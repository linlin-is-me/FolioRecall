"""One fixed Beijing Bailian provider with durable conservative cost accounting."""
import json
import asyncio
import os
from pathlib import Path
import sqlite3
import time
from urllib.parse import urlparse
import uuid

MODEL = 'qwen3-vl-plus-2025-12-19'


class TaskStopped(RuntimeError):
    pass


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

    def reserve(self, amount):
        with sqlite3.connect(self.path, timeout=5) as db:
            db.execute('BEGIN IMMEDIATE')
            used = db.execute('SELECT COALESCE(SUM(cny),0) FROM calls').fetchone()[0]
            if used + amount * 1.25 > self.cap:
                raise TaskStopped('phase_budget_exhausted')
            key = uuid.uuid4().hex
            db.execute('INSERT INTO calls VALUES (?,?,?,?,?)', (key, amount, 'reserved_unknown_charge', '{}', time.time()))
        return key

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
        key = self.ledger.reserve(reservation)
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
                # Do not include provider response bodies, request headers or images in errors.
                raise TaskStopped(f'api_http_{response.status_code}')
            result = response.json()
        except (httpx.TimeoutException, asyncio.TimeoutError) as exc:
            raise TaskStopped('api_timeout') from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise TaskStopped('api_transport_or_json_error') from exc
        usage = result.get('usage', {})
        cost = self.ledger.settle(key, usage)
        try:
            message = result['choices'][0]['message']
            message = {k:message[k] for k in ('role','content','tool_calls') if k in message}
        except (KeyError, IndexError, TypeError) as exc:
            raise TaskStopped('api_invalid_response') from exc
        return {'message': message, 'usage': usage, 'cost_cny': cost,
                'cost_status': 'estimated_from_usage_not_invoice', 'ledger_id': key,
                'response_model': result.get('model'),
                'seconds': time.monotonic()-started}
