"""Assistant-authored bilingual acceptance tasks grounded in the three JRC PDFs.

Reference facts were checked against extracted pages (chart tasks also against
rendered pages). They are not human gold labels or independent annotator reviews.
"""
import argparse
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from foliorecall.io import read_rows, write_rows, write_json, provenance


def task_specs():
    # category, English question, Chinese question, physical evidence IDs, key facts
    return [
('single_page','In the CCAM energy report, what three mechanisms determine the energy impact of automation?',
 'CCAM 能耗报告将自动化的能源影响归为哪三种机制？',['JRC141212:7'],
 ['Additional sensing/computation/communication and off-board energy; changes in propulsion energy via traffic dynamics; travel behaviour and modal shifts.']),
('single_page','Why can robotaxis increase urban energy use, and which shared services does the CCAM report contrast with them?',
 '为何无人出租车可能增加城市能耗？CCAM 报告用哪些共享服务与之对照？',['JRC141212:10'],
 ['Private robotaxi rides can shift travel from walking, cycling and transit and increase VKT; higher-occupancy shared shuttles and buses reduce VKT and better preserve active travel.']),
('single_page','What future city-level validation does the CCAM report propose?',
 'CCAM 报告计划在城市层面开展什么后续验证？',['JRC141212:12'],
 ['Replicate the study in a larger and more congested city to examine scalability and broader impacts.']),
('single_page','Which four technology domains classify the projects in the European zero-emission heavy-duty R&I review?',
 '欧洲零排放重型运输研发综述将项目分成哪四个技术领域？',['JRC141952:9'],
 ['Efficient vehicle and operations; zero-emission vehicle components; charging and refuelling infrastructure; vehicle integration and deployment.']),
('single_page','Which database and screening approach underlie the zero-emission heavy-duty project review?',
 '零排放重型运输项目综述采用哪个数据库和怎样的项目筛选方法？',['JRC141952:10'],
 ['TRIMIS, keyword searches followed by manual review; projects categorised by funding source, technology domain and readiness.']),
('single_page','Which battery-system improvements are reported in the conclusions of the heavy-duty R&I review?',
 '重型运输研发综述的结论报告了哪些电池系统改进？',['JRC141952:86'],
 ['Some projects achieved 20% higher energy density and 30% lower battery-pack weight; these are project achievements, not universal gains.']),
('single_page','What role does the stakeholder readiness report assign to battery-electric and hydrogen technologies?',
 '利益相关方技术成熟度报告分别如何定位纯电和氢能技术？',['JRC143054:7'],
 ['Battery electric is the primary pathway; hydrogen is complementary, especially for demanding long-haul use cases with BEV operational barriers.']),
('single_page','Why does the stakeholder study exclude carbon-containing alternative fuels?',
 '利益相关方研究为何排除含碳替代燃料？',['JRC143054:12'],
 ['Its scope is zero-tailpipe-emission technologies; carbon-containing fuels normally do not meet that boundary. Electricity and hydrogen are the two energy carriers considered.']),
('single_page','Which limitations qualify the stakeholder report\'s technology readiness scores?',
 '利益相关方报告的技术成熟度评分有哪些适用限制？',['JRC143054:13'],
 ['Scores reflect expressed stakeholder perceptions and the composition and size of the sample; energy-supply participation could have been stronger.']),
('single_page','What grid-related barrier does the stakeholder report identify beyond R&I?',
 '利益相关方报告指出了哪项研发之外的电网障碍？',['JRC143054:28'],
 ['Insufficient distribution-grid capacity and slow/costly grid connections can delay megawatt charging hubs; coordinated transport and energy infrastructure planning is needed.']),
('multi_page','Connect the CCAM report\'s classification of energy impacts with its proposed urban policy response.',
 '结合 CCAM 报告的能耗机制分类与城市政策建议，说明为什么技术节能仍需出行政策配合。',['JRC141212:7','JRC141212:8'],
 ['Automation adds system energy and can induce modal shifts; shared mobility, reduced car dependency and support for active travel address the behavioural component that hardware optimisation alone cannot remove.']),
('multi_page','How do the CCAM report\'s highway conclusions differ from its caution about mixed traffic with unstable ACC?',
 'CCAM 报告对高速公路节能的结论，与其对不稳定 ACC 混行环境的提醒有何联系？',['JRC141212:8','JRC141212:10'],
 ['Smoother flows can reduce energy use; unstable ACC-dominated networks may need a higher penetration of traffic-aware automated vehicles to stabilise flow. Benefits depend on traffic assumptions.']),
('multi_page','How does the CCAM report connect urban modal shifts to the need for validation in another city?',
 'CCAM 报告中的城市出行方式变化，为何需要在另一座城市继续验证？',['JRC141212:10','JRC141212:12'],
 ['Robotaxis and shared high-occupancy services affect VKT and active travel differently; the proposed larger, more congested city study tests scalability and broader impacts.']),
('multi_page','In the heavy-duty project review, which domain receives the largest funding share, and which bidirectional charging mechanism could let battery-electric trucks earn grid-related revenues?',
 '重型运输项目综述中，哪个领域获得最大资金份额？报告讨论的哪种双向充电机制可能让纯电卡车获得电网相关收入？',['JRC141952:36','JRC141952:85'],
 ['Vehicle integration and deployment receives the largest funding share; V2G can send electricity back to the grid and potentially earn reserve-market or arbitrage revenue.']),
('multi_page','Relate the heavy-duty review\'s infrastructure recommendations to examples of charging work in its conclusions.',
 '将重型运输综述的基础设施建议与结论中列出的充电项目工作联系起来。',['JRC141952:8','JRC141952:86'],
 ['It recommends high-power charging and smart grids; examples include modular high-power chargers, interoperability, megawatt charging and digital-twin/AI management tools.']),
('multi_page','What data collection period and follow-up workshop supported the stakeholder study, and what representation gap was noted?',
 '利益相关方研究的调查期和后续研讨会是什么时间？研究指出哪方面代表性不足？',['JRC143054:11','JRC143054:13'],
 ['Survey ran 10 December 2024 to 31 January 2025; workshop held 11 February 2025 in Brussels; stronger energy-supply representation would have helped.']),
('multi_page','How does the stakeholder study define long-haul and urban delivery use cases, and how do their adoption priorities differ?',
 '利益相关方研究如何定义长途运输和城市配送场景？二者的技术采用重点有何不同？',['JRC143054:12','JRC143054:8'],
 ['Long haul is 400+ km/day using heavy trucks; urban delivery is typically 50–150 km/day with 5–16t trucks. Long-haul priorities include range/refuelling speed; urban priorities include longevity/payload.']),
('multi_page','Combine the stakeholder report\'s electricity and hydrogen price expectations, retaining the units and comparison baselines.',
 '综合利益相关方报告的电价和氢价预期，保留单位及比较对象。',['JRC143054:31','JRC143054:32'],
 ['Electricity: 0.3 EUR/kWh or less versus diesel; hydrogen responses centre near 5 EUR/kg, qualitative competitive ranges 6–8 versus diesel and 2–4 versus electricity. These are respondent assessments.']),
('multi_page','Why should the stakeholder report\'s TCO projections not be read as guaranteed market outcomes?',
 '为何不能把利益相关方报告中的 TCO 预测视为保证实现的市场结果？',['JRC143054:13','JRC143054:32'],
 ['Findings are sample-dependent perceptions; projections depend on electricity/hydrogen prices, sufficient megawatt charging and industrialisation, and include individual respondents\' scenarios.']),
('multi_page','Relate the stakeholder report\'s preferred technology pathway to its identified grid deployment barrier.',
 '将利益相关方报告优先支持的技术路线与其识别的电网部署障碍联系起来。',['JRC143054:7','JRC143054:28'],
 ['Battery electric is prioritised, but grid capacity and costly/slow connections may hinder charging deployment, requiring coordinated infrastructure planning.']),
('cross_document','Compare the evidence-gathering methods of the heavy-duty R&I project review and the stakeholder readiness report.',
 '比较重型运输研发项目综述与利益相关方成熟度报告的证据收集方法。',['JRC141952:10','JRC143054:11','JRC143054:13'],
 ['Project review uses TRIMIS keyword screening and manual review; readiness study uses a stakeholder survey and workshop combining quantitative and qualitative evidence.']),
('cross_document','How do the heavy-duty R&I review and stakeholder report connect infrastructure and technology adoption?',
 '重型运输研发综述与利益相关方报告如何解释基础设施与技术采用的关系？',['JRC141952:8','JRC143054:28'],
 ['The review calls for high-power charging, hydrogen refuelling and smart grids; the stakeholder report identifies grid capacity and connection delays as concrete deployment barriers.']),
('cross_document','Contrast the heavy-duty R&I review\'s advanced-battery recommendation with the stakeholder report\'s technology priorities.',
 '对照重型运输研发综述的先进电池建议与利益相关方报告的技术优先级。',['JRC141952:8','JRC143054:7'],
 ['Review recommends improving battery and fuel-cell performance/cost/manufacturing; stakeholder report prioritises battery-electric development while retaining hydrogen for demanding long haul.']),
('cross_document','Why does automation not automatically deliver lower transport energy, and why does mature ZE-HDV technology not automatically guarantee adoption?',
 '为什么自动化不必然降低交通能耗，而较成熟的零排放重型车辆技术也不必然得到采用？',['JRC141212:7','JRC141212:8','JRC143054:8'],
 ['Automation has additional system loads and induced-travel risks; ZE-HDV adoption still depends on cost/TCO and use-case constraints such as range and refuelling.']),
('cross_document','Compare the system boundaries of the CCAM energy analysis and the stakeholder ZE-HDV definition.',
 '比较 CCAM 能源分析的系统范围与利益相关方研究对零排放重型车辆的定义范围。',['JRC141212:7','JRC143054:10','JRC143054:12'],
 ['CCAM includes sensing/computation/communications and off-board data centres as well as propulsion/behaviour; stakeholder study focuses on zero-tailpipe technologies and does not undertake detailed upstream energy-production analysis.']),
('cross_document','What does the CCAM report say about urban robotaxis, and how does the heavy-duty review describe transport automation achievements?',
 'CCAM 报告如何评价城市无人出租车的影响？重型运输综述又报告了哪些交通自动化进展？',['JRC141212:10','JRC141952:85'],
 ['Private robotaxis can increase VKT and energy through modal shifts; heavy-duty review describes coordination, platooning, automated driving and 5G/logistics platforms, without proving every automated service saves energy.']),
('cross_document','Compare the methodological limitations highlighted in the CCAM report\'s future work and the stakeholder readiness report.',
 '比较 CCAM 报告后续工作与利益相关方成熟度报告指出的方法局限。',['JRC141212:12','JRC143054:13'],
 ['CCAM seeks validation in a larger, more congested city; stakeholder results depend on sample size/composition and perceptions, with energy-supply underrepresentation.']),
('cross_document','How do the two heavy-duty reports describe the status of EU CO2 standards in their own text? Do not replace their wording with current law.',
 '两份重型运输报告各自在文中如何描述欧盟 CO2 标准的状态？请勿用现行法规替换报告原意。',['JRC141952:11','JRC143054:9'],
 ['R&I review describes standards under revision and a revision proposal; stakeholder report refers to adopted Regulation (EU) 2024/1610 and its targets. This is a contrast between report texts, not present legal advice.']),
('numeric_chart','In the stakeholder support chart, how many more R&I support counts go to battery electric than hydrogen?',
 '利益相关方支持度图中，纯电研发的支持计数比氢能多多少？',['JRC143054:8'],
 ['Battery electric 66, hydrogen 38, difference 28 support counts; these are counts of support for actions, not a percentage or necessarily unique respondents.']),
('numeric_chart','Using the stakeholder support chart, what share of battery-electric support counts concerns battery technology plus charging infrastructure?',
 '按利益相关方支持度图，电池技术与充电基础设施合计占纯电支持计数的多少？',['JRC143054:8'],
 ['27+24=51 out of 66, approximately 77.27%; denominator is support counts within the battery-electric domain.']),
('numeric_chart','How many complete unique survey responses and SME/start-up/scale-up respondents are reported, and what share does the latter represent?',
 '调查报告了多少份完整独立回复，其中中小企业、初创或成长企业有多少，占比多少？',['JRC143054:14'],
 ['65 complete unique responses; 14 in the grouped SME/start-up/scale-up category; report rounds the proportion to 21% (14/65 about 21.54%).']),
('numeric_chart','According to the respondent chart, what is the combined count for vehicle OEMs, parts/technology suppliers and research/academia?',
 '按受访者类型图，整车厂、零部件或技术供应商、研究与学术机构合计多少份回复？',['JRC143054:14'],
 ['9+15+13=37 responses.']),
('numeric_chart','Which electricity-price interval has the highest response frequency in Figure 18, and what upper price does the accompanying text identify?',
 '图18中哪个电价区间的回复频率最高？正文给出的价格上限是多少？',['JRC143054:31'],
 ['The 0.25–0.30 EUR/kWh interval has the highest bar; text says 0.3 EUR/kWh or less.']),
('numeric_chart','Which hydrogen-price interval has the highest response frequency in Figure 19, and around what price do answers centre?',
 '图19中哪个氢价区间的回复频率最高？回复大致围绕什么价格？',['JRC143054:32'],
 ['4–6 EUR/kg is the highest interval, consistent with answers centring around 5 EUR/kg.']),
('numeric_chart','For the individual manufacturer scenario in the stakeholder report, calculate the eight-year TCO difference at 100,000 km per year between 0.74 and 1.33 EUR/km.',
 '按利益相关方报告中某厂商的情景，以每年10万公里、8年运营计算，0.74与1.33欧元/公里的 TCO 相差多少？',['JRC143054:32'],
 ['Diesel minus electric: (1.33−0.74)×100000×8 = 472000 EUR; it is one manufacturer projection, not a general observed saving.']),
('numeric_chart','The heavy-duty R&I review reports 72 zero-emission-component projects out of 153 reviewed projects. What percentage is that?',
 '重型运输研发综述中的153个项目有72个属于零排放车辆部件，占比多少？',['JRC141952:85'],
 ['72/153×100 = approximately 47.06%.']),
('insufficient','What exact orbital period of Saturn do these three transport reports establish?',
 '这三份交通报告证明的土星精确公转周期是多少？',[],[]),
('insufficient','Name every organisation that submitted each of the 65 individual stakeholder survey responses.',
 '请给出65份利益相关方调查回复各自对应的全部机构名称。',['JRC143054:13'],
 ['The report aggregates anonymous responses by stakeholder type; individual respondent identities are not established.']),
('insufficient','Give the exact realised 2030 electricity price paid by every European long-haul truck operator, as measured by these reports.',
 '请根据报告给出2030年每一家欧洲长途货运企业实际支付的精确电价。',[],
 ['The reports contain assessments/projections, not a complete set of realised operator-level 2030 prices.']),
('insufficient','Identify which of the surveyed companies will certainly become insolvent in 2030 and their exact insolvency dates.',
 '请指出受访企业中哪些必然在2030年破产，并给出各自的准确破产日期。',[],[]),
]


def freeze(corpus, output):
    from foliorecall.retrieval import page_identity
    output=Path(output)
    if output.exists():
        raise ValueError('冻结目录已存在，请使用新版本')
    pages=read_rows(Path(corpus)/'pages.jsonl')
    ids={p['page_id'] for p in pages}
    tasks=[]
    for i,(category,en,zh,evidence,facts) in enumerate(task_specs(),1):
        if not set(evidence).issubset(ids):
            raise ValueError('参考证据页缺失')
        tasks.append({'task_id':f'jrc-{i:02d}','category':category,'question_en':en,'question_zh':zh,
            'answerable':category!='insufficient','reference_facts':facts,'evidence_page_ids':evidence,
            'rubric':{'fact_correctness':'Each asserted factual statement must agree with source and preserve scope/units; score supported assertions / all assertions.',
                      'answer_coverage':'Count supplied reference fact components / requested components; no-answer tasks require explicit insufficiency.',
                      'citation_support':'Each claim must be supported by its cited page content, not just a valid page ID.',
                      'insufficiency_handling':'Do not invent missing details; distinguish predictions, respondent views and observed facts.'},
            'review':{'author':'assistant','assistant':'source_text_checked',
                      'chart_review':'rendered pages 8, 14, 31, 32 of JRC143054 checked' if category=='numeric_chart' and i<=34 else 'not_applicable',
                      'human':'not_checked'},
            'scope':'assistant-authored acceptance reference, not human gold; alternative supporting pages allowed after review, never used to tune current run'})
    counts=dict(Counter(t['category'] for t in tasks))
    assert counts=={'single_page':10,'multi_page':10,'cross_document':8,'numeric_chart':8,'insufficient':4}
    provenance(output,{'corpus':str(corpus),'counts':counts,'source':'three fixed JRC reports','author':'assistant'})
    write_rows(output/'tasks.jsonl',tasks)
    write_json(output/'status.json',{'split':'acceptance','logical_tasks':40,'language_variants':80,
        'qa_reference_frozen':True,'evaluation_allowed':True,'human_gold':False,
        'corpus_pages':page_identity(pages),
        'human_review_selection':['jrc-01','jrc-04','jrc-11','jrc-16','jrc-21','jrc-25','jrc-29','jrc-33','jrc-37','jrc-38'],
        'scope':'Initial assistant reference freeze; human review pending. No system outputs used in task construction.'})
    print(counts)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--corpus',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    freeze(args.corpus,args.output)
