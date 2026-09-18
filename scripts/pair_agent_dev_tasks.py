"""Attach assistant translations without upgrading unverified upstream answers."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from foliorecall.io import read_rows, write_rows, read_json, write_json, provenance

TRANSLATIONS = {
'hr:0':'2018至2023年临时合同使用的变化，如何影响跨国流动的欧盟劳动者与东道国公民之间的就业稳定性差距？',
'hr:120':'欧盟要在2030年达到2000万名 ICT 专家的目标，ICT 专家就业人数每年需要增长百分之多少？',
'hr:103':'分析绿色转型如何影响欧盟转型行业的劳动力市场动态。',
'hr:109':'奥地利、法国与捷克、瑞典的区域收入趋同趋势有何不同？哪些因素可以解释这些差异？',
'hr:1':'是什么推动了德国等欧盟国家女性劳动参与率上升？',
'hr:10':'比较欧洲各国代金券计划与税收激励在减少非正规家政劳动方面的效果。',
'hr:106':'提高未充分就业群体的劳动参与率，能否完全抵消到2040年预计每年减少200万劳动者的影响？',
'hr:11':'区域绿色转型脆弱性指数是否表明：工业结构相近时，依赖污染行业的富裕地区比贫困地区面临更高风险？',
'computer_science:103':'一个 DataFrame 列的100行中有5个缺失值，使用 fillna() 后缺失数据占比是多少？',
'computer_science:24':'文档明确列出了几个用于日期处理和图形界面开发的 Python 内置模块示例？',
'computer_science:10':'按照文档说明，描述 TCP 三次握手的步骤顺序。',
'computer_science:114':'二分查找如何达到 O(log N) 的效率？',
'computer_science:0':'根据文档中的命名约定，Python 变量名可以以数字开头吗？',
'computer_science:1':'瀑布模型与 V 模型在软件开发生命周期中整合测试的方式有何主要区别？',
'computer_science:11':'Python 函数名以下划线开头有什么作用？这会如何影响对模块调用 help() 时显示的内容？',
'computer_science:118':'在 Pandas 中，如果不传入参数，tail(n) 会返回末尾的 n 行吗？',
'finance_en:101':'富国银行2024年相较2023年的贷款净核销额增加了百分之多少？',
'finance_en:104':'花旗集团2020至2024年的其他自有房地产资产减少了多少？',
'finance_en:10':'摩根士丹利2024年的收入增长与费用效率比率之间有什么关系？',
'finance_en:14':'概述高盛2024年 Form 10-K 披露的主要业务分部。',
'finance_en:0':'截至年末，摩根大通是否已执行其300亿美元股票回购计划的一半以上？',
'finance_en:1':'美国银行 B 系列优先股与花旗 O 系列优先股的股息机制和投票权有何不同？',
'finance_en:100':'摩根士丹利2024年的平均管理 VaR 总额是多少？',
'finance_en:102':'花旗证券融资交易中，有多少负债面临潜在的司法管辖权不确定性？',
'industrial:121':'按美国空军技术手册与指令，服装、鞋套和擦拭布的包装方式有什么区别？',
'industrial:148':'Hastelloy R-235 镍基耐蚀合金的钛含量百分比范围是多少？',
'industrial:34':'技术手册如何在不同军事领域兼顾程序一致性和操作灵活性，同时保持对标准化规程的遵守？',
'industrial:83':'为极寒环境设计压力系统的主要挑战是什么？材料性能如何影响这些设计？',
'industrial:0':'紧急停机程序涉及哪些部件？它们如何确保系统隔离？',
'industrial:1':'说明飞机加油作业中实施紧急停机所需的动作顺序。',
'industrial:10':'文档讨论的各类电镀方法在航空航天部件中分别有哪些具体用途？',
'industrial:108':'按美国空军技术手册，ATS 在 ICT 操作中必须穿什么背心？',
'pharmaceuticals:106':'计算 FDA 2015财年 ANDA 审查完成率比原定绩效目标提高了多少个百分点。',
'pharmaceuticals:161':'2019年与2004年的生物制品研究性新药申请数量分别是多少？',
'pharmaceuticals:102':'2017年的 ANDA 提交模式与2018年的监管行动有何联系？这反映了怎样的 FDA 审查周期？',
'pharmaceuticals:111':'现有医疗支付体系可能如何促成美国阿片类药物滥用增加？',
'pharmaceuticals:0':'哪一年通过的 Kefauver-Harris 修正案使随机对照试验成为评价临床疗效的标准？',
'pharmaceuticals:10':'说明生物类似药审批的分层证据结构如何排列各类研究，以及这对临床应用意味着什么。',
'pharmaceuticals:1':'FDA 在疫情激增前及期间的会议响应时间有何变化？',
'pharmaceuticals:103':'指出文中 FDA 对药物配制设施开展的检查类别及各类检查数量。',
}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--selection',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    output=Path(args.output)
    if output.exists():
        raise ValueError('输出已存在')
    rows=read_rows(Path(args.selection)/'tasks.jsonl')
    if {r['task_id'] for r in rows}!=set(TRANSLATIONS):
        raise ValueError('翻译与冻结任务 ID 不一致')
    for row in rows:
        row['question_zh']=TRANSLATIONS[row['task_id']]
        row['review']['translation']='assistant_translated; human not checked'
    provenance(output,{'selection':args.selection,'scope':'paired questions only; upstream answer review still pending'})
    write_rows(output/'tasks.jsonl',rows)
    write_json(output/'status.json',{'split':'development','selection_frozen':True,'logical_tasks':40,'language_variants':80,
        'qa_reference_frozen':False,'evaluation_allowed':False,
        'blocking':['verify upstream answers against source pages; resolve ambiguous task categories']})
