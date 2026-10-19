"""Presentation shared by the local retrieval and document-chat views."""
import re


def document_label(name, limit=46):
    """A display-only filename label; records and citation IDs stay untouched."""
    text = str(name).replace('\\', '/').rsplit('/', 1)[-1]
    text = re.sub(r'\.(pdf|png|jpe?g|webp)$', '', text, flags=re.I)
    # EU Publications Office filenames append a catalogue code to the title.
    text = re.sub(r'[-_][A-Z]{2}\d{7}[A-Z]{3}$', '', text)
    text = ' '.join(text.replace('_', ' ').split())
    text = text[:1].upper() + text[1:]
    if len(text) > limit:
        prefix = text[:limit - 1]
        boundary = prefix.rfind(' ')
        text = (prefix[:boundary] if boundary > limit // 2 else prefix) + '…'
    return text

EMPTY_ANSWER = '''<div class="fr-empty">
<span class="fr-eyebrow">答案与证据</span><h3>从一个文档问题开始</h3>
<p>输入问题，或点击示例填入后提交。文档研究按责任取证、分析与核验，单Agent适合快速问答，RAG直接使用检索前页。</p>
<p>回答后可用中文追问；点击引用查看原页。换话题时新建会话。候选包中的 showcase/index.html 可离线查看真实案例，无需API。</p></div>'''

BUSY_ANSWER = '''<div class="fr-empty" role="status">
<span class="fr-eyebrow">正在执行</span><h3>任务正在处理</h3>
<p>上方展示实际进度；研究模式依次规划、取证、分析与核验，必要时返工一次。结束后在此查看答案、原页与核验记录。</p></div>'''

CSS = '''
.gradio-container {
  --fr-surface: #ffffff; --fr-canvas: #f5f7f6; --fr-ink: #203732;
  --fr-muted: #5c706a; --fr-line: #dde6e2; --fr-accent: #176a58;
  --fr-tint: #edf5f1; --fr-warn: #8b5b20;
  max-width: 1440px !important; margin: auto !important;
  padding: clamp(16px, 2.5vw, 36px) !important; background: var(--fr-canvas) !important;
}
.gradio-container .main { padding: 0 !important; }
.dark .gradio-container {
  --fr-surface: #192622; --fr-canvas: #111c18; --fr-ink: #e2ece7;
  --fr-muted: #b0c1b9; --fr-line: #33463c; --fr-accent: #8fd1b8;
  --fr-tint: #243c31; --fr-warn: #e4bd7d;
}
.fr-header { display: flex; align-items: center; justify-content: space-between;
  gap: 24px; padding: 8px 0 24px; color: var(--fr-ink); }
.fr-brand { display: flex; align-items: center; gap: 16px; min-width: 0; }
.fr-brand > div { min-width: 0; }
.fr-wordmark { display: block; width: 232px; height: auto; object-fit: contain; }
.dark .fr-wordmark { background: #f5f7f6; border-radius: 6px; }
.fr-brand h1 { display: flex; align-items: center; margin: 0 !important; font-size: 27px !important; letter-spacing: -.7px; font-weight: 650; }
.fr-brand h1 span { font-size: 18px; font-weight: 450; margin-left: 10px; color: var(--fr-muted); }
.fr-brand p { margin: 5px 0 0; font-size: 14px; color: var(--fr-muted); }
.fr-badges { display: flex; gap: 8px; flex-wrap: wrap; }
.fr-badge { border: 1px solid var(--fr-line); border-radius: 20px; padding: 6px 11px;
  font-size: 12px; color: var(--fr-muted); background: var(--fr-surface); }
.fr-library { display: flex; flex-wrap: wrap; align-items: baseline; gap: 6px 16px;
  padding: 0 0 16px; font-size: 12px; color: var(--fr-muted); }
.fr-library strong { color: var(--fr-ink); font-weight: 550; }
.fr-library span { overflow-wrap: anywhere; }
#workspace-tabs .tab-container { gap: 6px; padding-bottom: 10px; }
#workspace-tabs .tab-container button { padding: 10px 22px; font-weight: 600; border-radius: 9px; }
#workspace-tabs .tab-container button.selected { background: var(--fr-tint); color: var(--fr-accent); border: 0; }
#workspace-tabs > .tabitem { padding: 22px 0 0; border: 0; }
.fr-workspace { gap: 24px !important; align-items: flex-start; }
.fr-controls { background: var(--fr-surface); border: 1px solid var(--fr-line) !important;
  border-radius: 16px; padding: 22px !important; gap: 18px !important; }
.fr-results { gap: 16px !important; min-width: 0 !important; }
.fr-section h2 { color: var(--fr-ink); font-size: 20px !important; margin: 0 0 6px !important; }
.fr-section p, .fr-note p { color: var(--fr-muted); font-size: 13px; line-height: 1.7; }
.fr-controls textarea { line-height: 1.7 !important; }
.fr-controls button.primary { min-height: 44px; border-radius: 10px; font-weight: 600;
  background: #176a58; color: #fff; }
.fr-controls .form { border: 0; box-shadow: none; }
.fr-controls .dataset { border: 0; }
.fr-controls .dataset button { text-align: left; white-space: normal; line-height: 1.5; }
#search-gallery { background: var(--fr-surface); border: 1px solid var(--fr-line); border-radius: 14px; }
#search-status { color: var(--fr-muted); font-size: 12px; }
#search-table { overflow-x: auto; }
#runtime-info { margin-top: 22px; border-top: 1px solid var(--fr-line); }
#runtime-info .prose { overflow-wrap: anywhere; font-size: 12px; }
#conversation { border-radius: 12px; border-color: var(--fr-line); }
#answer-panel { color: var(--fr-ink); }
#answer-panel:empty, #progress-panel:empty { display: none; }
.fr-research-log { border: 1px solid var(--fr-line); border-radius: 10px; padding: 12px 16px; background: var(--fr-surface); }
.fr-research-log > summary { cursor: pointer; color: var(--fr-muted); font-size: 13px; }
.fr-answer, .fr-empty { background: var(--fr-surface); border: 1px solid var(--fr-line);
  padding: 26px; border-radius: 16px; }
.fr-empty { min-height: 240px; display: flex; flex-direction: column; justify-content: center; }
.fr-empty h3 { font-size: 23px; margin: 16px 0 10px; font-weight: 550; }
.fr-empty p { color: var(--fr-muted); line-height: 1.8; max-width: 420px; }
.fr-eyebrow { color: var(--fr-accent); font-size: 12px; font-weight: 650; letter-spacing: .06em; }
.fr-answer-head { display: flex; justify-content: space-between; align-items: center;
  flex-wrap: wrap; gap: 8px; margin-bottom: 20px; }
.fr-answer-head h3 { margin: 0; font-size: 20px; }
.fr-answer-status { color: var(--fr-accent); background: var(--fr-tint); font-size: 12px;
  padding: 5px 10px; border-radius: 20px; }
.fr-answer-status.warning { color: var(--fr-warn); }
.fr-claim { padding: 0 0 20px; margin-bottom: 20px; border-bottom: 1px solid var(--fr-line); }
.fr-claim:last-child { border-bottom: 0; margin-bottom: 0; padding-bottom: 0; }
.fr-claim p { font-size: 15px; line-height: 1.85; margin: 0 0 12px; white-space: pre-wrap; overflow-wrap: anywhere; }
.fr-citations { display: flex; gap: 6px; flex-wrap: wrap; }
.fr-citation { color: var(--fr-accent) !important; background: var(--fr-tint); font-size: 12px;
  border: 1px solid var(--fr-line); border-radius: 6px; padding: 5px 9px; text-decoration: none !important;
  max-width: 100%; overflow-wrap: anywhere; line-height: 1.6; }
.fr-citation:hover { text-decoration: underline !important; }
.fr-citation:focus-visible, .fr-evidence summary:focus-visible { outline: 2px solid var(--fr-accent); outline-offset: 3px; }
.fr-unresolved { border-left: 3px solid var(--fr-warn); padding: 2px 0 2px 16px; margin: 18px 0 0; }
.fr-unresolved h4 { margin: 0 0 8px; font-size: 14px; color: var(--fr-warn); }
.fr-unresolved p { line-height: 1.8; font-size: 14px; margin: 6px 0; white-space: pre-wrap; }
.fr-evidence-list { margin-top: 24px; }
.fr-evidence-list > h3 { font-size: 16px; margin: 0 0 12px; }
.fr-evidence { border: 1px solid var(--fr-line); border-radius: 10px; margin-top: 8px; overflow: hidden; }
.fr-evidence summary { cursor: pointer; padding: 14px 16px; font-size: 13px; overflow-wrap: anywhere; background: var(--fr-tint); }
.fr-evidence figure { margin: 0; padding: 16px; scroll-margin-top: 24px; }
.fr-evidence figcaption { color: var(--fr-muted); font-size: 12px; margin-bottom: 12px; overflow-wrap: anywhere; }
.fr-evidence img { display: block; max-width: 100%; height: auto; margin: 0 auto; }
#agent-progress { border: 1px solid var(--fr-line); background: var(--fr-surface); border-radius: 12px; padding: 16px 20px; color: var(--fr-ink); }
#agent-progress h3 { font-size: 14px; margin: 0 0 12px; }
.fr-progress-head { display: flex; justify-content: space-between; flex-wrap: wrap; gap: 8px; }
.fr-progress-head span { font-size: 12px; color: var(--fr-accent); }
#agent-progress ol { max-height: 280px; overflow-y: auto; margin: 0; padding: 0; list-style: none; }
#agent-progress li { border-left: 2px solid var(--fr-line); padding: 4px 0 12px 14px; margin-left: 3px; }
#agent-progress li:last-child { border-color: var(--fr-accent); padding-bottom: 4px; }
#agent-progress strong { font-size: 13px; }
#agent-progress small { display: block; color: var(--fr-muted); font-size: 12px; line-height: 1.6; overflow-wrap: anywhere; margin-top: 5px; }
.fr-step-status { float: right; font-size: 11px; color: var(--fr-muted); }
.fr-step-status.failed, .fr-step-status.error, .fr-step-status.limit { color: var(--fr-warn); }
@media (max-width: 800px) {
  .fr-header { align-items: flex-start; flex-direction: column; gap: 14px; padding-bottom: 18px; }
  .fr-brand h1 { font-size: 23px !important; }
  .fr-wordmark { width: min(232px, 58vw); }
  .fr-workspace { flex-direction: column !important; gap: 18px !important; }
  .fr-controls, .fr-results { width: 100% !important; min-width: 0 !important; }
  .fr-controls { padding: 18px !important; }
  .fr-answer, .fr-empty { padding: 20px; }
  #search-gallery { height: 480px !important; }
}
'''
