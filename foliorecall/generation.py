"""One fixed Beijing Bailian provider with durable conservative cost accounting."""
import json
import math
import asyncio
import os
import re
from pathlib import Path
import sqlite3
import time
from urllib.parse import urlparse
import uuid
from email.utils import parsedate_to_datetime
from datetime import timezone

MODEL = 'qwen3-vl-plus-2025-12-19'


def retry_after_metadata(headers):
    """Freeze a service delay at receipt; absolute deadlines are diagnostic only."""
    value = headers.get('Retry-After')
    if value is None:
        return {'retry_after_status': 'absent'}
    value = value.strip()
    if re.fullmatch(r'[0-9]+', value):
        try:
            seconds = float(value)
            if math.isfinite(seconds):
                return {'retry_after_status': 'resolved', 'retry_after_source': 'delay_seconds',
                        'retry_after_seconds': seconds, 'retry_not_before': time.time()+seconds}
        except (ValueError, OverflowError):
            pass
        return {'retry_after_status': 'invalid'}
    def timestamp(raw):
        parsed = parsedate_to_datetime(raw)
        return parsed.replace(tzinfo=parsed.tzinfo or timezone.utc).timestamp()
    try:
        deadline = timestamp(value)
    except (ValueError, TypeError, OverflowError, IndexError):
        return {'retry_after_status': 'invalid'}
    out = {'retry_after_status': 'unresolved', 'retry_after_source': 'http_date',
           'retry_not_before': deadline}
    try:
        server_now = timestamp(headers.get('Date'))
    except (ValueError, TypeError, OverflowError, IndexError):
        return out
    out.update(retry_after_status='resolved', retry_after_source='http_date_server_date',
               retry_after_seconds=max(0., deadline-server_now))
    return out


def diagnostic_text(content, secret=None, limit=32768):
    """Bound only visible output; never serialize the enclosing provider response."""
    if not isinstance(content, str):
        return {'content_type':type(content).__name__, 'text':None,
                'original_length':None, 'truncated':False}
    value = content
    for key in (secret, os.environ.get('DASHSCOPE_API_KEY')):
        if key:
            value = value.replace(key, '[REDACTED]')
            value = value.replace(json.dumps(key)[1:-1], '[REDACTED]')
    value = re.sub(r'<think\b[^>]*>.*?(?:</think>|$)', '[REDACTED_REASONING]', value, flags=re.I|re.S)
    value = re.sub(r'"reasoning_content"\s*:\s*"(?:\\.|[^"\\])*"',
                   '"reasoning_content":"[REDACTED]"', value, flags=re.I)
    value = re.sub(r'data:[^\s"\']*;base64,[A-Za-z0-9+/=\s]+', '[REDACTED_IMAGE]', value, flags=re.I)
    value = re.sub(r'\b[A-Za-z0-9+/]{256,}={0,2}', '[REDACTED_ENCODED_DATA]', value)
    value = re.sub(r'(?i)\bsk-[A-Za-z0-9_.\\-]+', '[REDACTED_KEY]', value)
    value = re.sub(r'(?i)\bBearer\s+[^\s"\']+', 'Bearer [REDACTED]', value)
    value = re.sub(r'(?im)\bAuthorization\s*:[^\r\n]*', 'Authorization: [REDACTED]', value)
    return {'content_type':'str','text':value[:limit], 'original_length':len(content),
            'sanitized_length':len(value),'truncated':len(value)>limit}


def http_error_diagnostic(response, secret):
    """Keep a small error-field whitelist, never raw bodies or request headers."""
    out = {'http_status':response.status_code,
           'content_type':diagnostic_text(response.headers.get('content-type',''), secret, 128)['text']}
    try:
        body = response.json()
    except ValueError:
        return out
    if not isinstance(body,dict):
        return out
    error = body.get('error',body)
    if isinstance(error,dict):
        for name in ('code','type','message'):
            if isinstance(error.get(name),str):
                limit = 2048 if name == 'message' else 256
                value = diagnostic_text(error[name],secret,limit)
                out[name] = value['text']
                if name == 'message':
                    out['message_original_length'] = value['original_length']
                    out['message_truncated'] = value['truncated']
    request_id = body.get('request_id')
    if not isinstance(request_id,str) and isinstance(error,dict):
        request_id = error.get('request_id')
    if isinstance(request_id,str):
        out['request_id'] = diagnostic_text(request_id,secret,256)['text']
    return out


class TaskStopped(RuntimeError):
    def __init__(self, reason, record=None):
        super().__init__(reason)
        self.record = record or {}


def validate_agent_config(config):
    if config.get('provider') != 'bailian' or config.get('model') != MODEL or config.get('region') != 'cn-beijing':
        raise ValueError('仅支持北京百炼 qwen3-vl-plus-2025-12-19，不能自动替换模型或地域')
    if config.get('temperature') != 0 or config.get('enable_thinking') is not False:
        raise ValueError('必须关闭思考并设置 temperature=0')
    bounds = {'max_searches': 3, 'max_pages': 6, 'max_requests': 8,
              'max_output_tokens': 8000, 'request_timeout': 60, 'task_timeout': 180,
              'max_pixels': 1310720}
    for key, maximum in bounds.items():
        if type(config.get(key)) is not int or not 1 <= config[key] <= maximum:
            raise ValueError(f'{key} 必须为 1–{maximum} 的整数')
    if not isinstance(config.get('phase_budget_cny'), (int,float)) or not 0 < config['phase_budget_cny'] <= 150:
        raise ValueError('阶段 A 的 API 上限不能超过 150 元')
    if not config.get('ledger'):
        raise ValueError('缺少持久费用账本路径')


def read_agent_config(path):
    from .io import read_json
    config = read_json(path)
    validate_agent_config(config)
    config['ledger'] = str((Path(path).resolve().parent / config['ledger']).resolve())
    return config


class CostLedger:
    """Reservations survive crashes and count as estimates until reconciled."""
    def __init__(self, path, cap):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path, self.cap = str(path), cap
        with sqlite3.connect(self.path) as db:
            db.execute('CREATE TABLE IF NOT EXISTS calls (id TEXT PRIMARY KEY, cny REAL, status TEXT, usage TEXT, at REAL)')
            columns = {r[1] for r in db.execute('PRAGMA table_info(calls)')}
            for name, kind in [('reservation_cny','REAL'), ('context','TEXT'), ('error','TEXT')]:
                if name not in columns:
                    db.execute(f'ALTER TABLE calls ADD COLUMN {name} {kind}')

    def reserve(self, amount, context=None):
        with sqlite3.connect(self.path, timeout=5) as db:
            db.execute('BEGIN IMMEDIATE')
            used = db.execute('SELECT COALESCE(SUM(cny),0) FROM calls').fetchone()[0]
            if used + amount * 1.25 > self.cap:
                raise TaskStopped('phase_budget_exhausted')
            key = uuid.uuid4().hex
            db.execute('INSERT INTO calls (id,cny,status,usage,at,reservation_cny,context) VALUES (?,?,?,?,?,?,?)',
                       (key, amount, 'reserved_unknown_charge', '{}', time.time(), amount, json.dumps(context or {})))
        return key

    def fail(self, key, reason):
        with sqlite3.connect(self.path) as db:
            db.execute('UPDATE calls SET error=? WHERE id=?', (reason, key))

    def records(self):
        with sqlite3.connect(self.path) as db:
            db.row_factory = sqlite3.Row
            return [dict(row) for row in db.execute('SELECT * FROM calls ORDER BY at,id')]

    def settle(self, key, usage):
        prompt, output = usage.get('prompt_tokens'), usage.get('completion_tokens')
        if type(prompt) is not int or type(output) is not int or prompt < 0 or output < 0:
            raise TaskStopped('missing_usage')
        rate = 1 if prompt <= 32768 else 1.5 if prompt <= 131072 else 3
        estimated = (prompt * rate + output * rate * 10) / 1e6
        with sqlite3.connect(self.path) as db:
            db.execute('UPDATE calls SET cny=?, status=?, usage=? WHERE id=?',
                       (estimated, 'estimated_from_usage', json.dumps(usage), key))
        return estimated

    def total(self):
        with sqlite3.connect(self.path) as db:
            return db.execute('SELECT COALESCE(SUM(cny),0) FROM calls').fetchone()[0]


class Bailian:
    def __init__(self, config):
        validate_agent_config(config)
        self.config = config
        self.key = os.environ.get('DASHSCOPE_API_KEY')
        self.base = os.environ.get('DASHSCOPE_BASE_URL', '').rstrip('/')
        if not self.key or not self.base:
            raise ValueError('请在本机设置 DASHSCOPE_API_KEY 与北京地域 DASHSCOPE_BASE_URL；不要把密钥写入配置或日志')
        url = urlparse(self.base)
        allowed = url.hostname == 'dashscope.aliyuncs.com' or bool(url.hostname and url.hostname.endswith('.cn-beijing.maas.aliyuncs.com'))
        if url.scheme != 'https' or not allowed or url.username or url.password or url.query or url.fragment or url.path != '/compatible-mode/v1':
            raise ValueError('需要北京百炼 HTTPS compatible-mode/v1 端点')
        self.ledger = CostLedger(config['ledger'], config['phase_budget_cny'])

    def complete(self, messages, tools, max_tokens, timeout):
        import httpx
        # Highest published Beijing price band, 256k input and requested output.
        reservation = (262144 * 3 + max_tokens * 30) / 1e6
        key = self.ledger.reserve(reservation, getattr(self, 'context', {}))
        record = {'ledger_id':key, 'reservation_cny':reservation, 'usage':None,
                  'cost_status':'reserved_unknown_charge', 'max_tokens':max_tokens}
        payload = {'model': MODEL, 'messages': messages, 'temperature': 0,
                   'enable_thinking': False, 'max_tokens': max_tokens, 'stream': False,
                   'vl_high_resolution_images': False}
        if tools:
            payload['tools'] = tools
            payload['parallel_tool_calls'] = False
        else:
            payload['response_format'] = {'type': 'json_object'}
        started = time.monotonic()
        async def request():
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
                return await client.post(self.base+'/chat/completions', json=payload,
                    headers={'Authorization': 'Bearer '+self.key})
        async def bounded_request():
            return await asyncio.wait_for(request(), timeout=timeout)
        try:
            response = asyncio.run(bounded_request())
            if response.status_code != 200:
                record['http_diagnostic'] = http_error_diagnostic(response,self.key)
                record.update(retry_after_metadata(response.headers))
                raise TaskStopped(f'api_http_{response.status_code}')
            try:
                result = response.json()
            except ValueError as exc:
                raise TaskStopped('api_response_json_error') from exc
            if not isinstance(result, dict) or not isinstance(result.get('usage'), dict):
                raise TaskStopped('missing_usage')
            usage = result['usage']
            cost = self.ledger.settle(key, usage)
            record.update(usage=usage, cost_cny=cost, cost_status='estimated_from_usage_not_invoice')
            choice = result['choices'][0]
            if not isinstance(choice,dict):
                raise TaskStopped('api_invalid_response')
            record.update(response_id=result.get('id') if isinstance(result.get('id'),str) else None,
                          response_model=result.get('model') if isinstance(result.get('model'),str) else None,
                          finish_reason=choice.get('finish_reason') if isinstance(choice.get('finish_reason'),str) else None)
            message = choice['message']
            if not isinstance(message, dict) or not ('content' in message or 'tool_calls' in message):
                raise TaskStopped('api_invalid_response')
            message = {k:message[k] for k in ('role','content','tool_calls') if k in message}
        except (TaskStopped, httpx.HTTPError, asyncio.TimeoutError, KeyError, IndexError, TypeError) as exc:
            reason = str(exc) if isinstance(exc, TaskStopped) else (
                'api_timeout' if isinstance(exc, (httpx.TimeoutException, asyncio.TimeoutError)) else
                'api_transport_error' if isinstance(exc, httpx.HTTPError) else 'api_invalid_response')
            record['seconds'] = time.monotonic()-started
            self.ledger.fail(key, reason)
            raise TaskStopped(reason, record) from exc
        return record | {'message':message, 'seconds':time.monotonic()-started}
