import copy
import unittest

from foliorecall.research_display import research_html, view_state


def tasks():
    return [{'id':'t1','name':'取数','question':'Read <values>', 'requirement_ids':['r1'], 'depends_on':[]},
            {'id':'t2','name':'计算','question':'Compute ratio','requirement_ids':['r2'],'depends_on':['t1']}]


def saved_result():
    return {'completed':True, 'answer':{'claims':[{'text':'new','evidence_page_ids':['d:1']}],
            'insufficient_evidence':False,'unresolved':[]}, 'timing':{'task_seconds':17},
            'research':{'tasks':tasks(), 'requirements':[{'id':'r1','text':'Extract values'}],
                'workers':{'worker-1':{'task':tasks()[0],'done':True,'read_ids':['d:1']},
                           'worker-2':{'task':tasks()[1],'done':True,'read_ids':['d:1']}},
                'pages':[{'page_id':'d:1','document_name':'<report>.pdf','page_number':1,
                          'received_by':['worker-1','worker-2'],'analysis_versions':[0,1]}],
                'drafts':[{'version':0,'numbered_claims':[{'id':'v0-c1','text':'old'}]},
                          {'version':1,'numbered_claims':[{'id':'v1-c1','text':'new'}]}],
                'verifications':[{'version':1,'claims':[{'id':'v1-c1','supported':True,'reason':'new support'}],'issues':[]},
                                 {'version':0,'claims':[{'id':'v0-c1','supported':False,'reason':'old rejected'}],'issues':[]}],
                'rework':{'triggered':True,'retrieval':True,'task_id':'t2'},'verification_status':'passed'}}


class ResearchDisplayTests(unittest.TestCase):
    def test_render_preserves_input_and_binds_explicit_versions(self):
        result=saved_result(); before=copy.deepcopy(result)
        value=research_html(result)
        self.assertEqual(result,before)
        self.assertIn('&lt;values&gt;',value)
        self.assertNotIn('<report>',value)
        self.assertIn('qa-page-d%3A1',value)
        self.assertIn('成功读页的取证任务',value)
        self.assertNotIn('收到原页的取证角色',value)
        self.assertIn('不代表取证模型随后已看到页面图像',value)
        self.assertIn('分析收到（版本）',value)
        self.assertIn('最终引用',value)
        self.assertIn('已引用',value)
        self.assertNotIn('草稿采用',value)
        self.assertIn('取数、计算',value)
        self.assertLess(value.index('v1-c1'),value.index('new support'))
        self.assertLess(value.index('old rejected'),value.index('v1-c1'))
        self.assertIn('首次核验（返工前） · v0',value)
        self.assertIn('末次核验 · v1',value)
        self.assertIn('负责人：计算',value)

    def test_live_interleaving_rework_and_blocked(self):
        events=[{'event':'coordinate','status':'completed','tasks':tasks(),'requirements':[]},
                {'event':'research_task','task_id':'t1','state':'completed','phase':'main'},
                {'event':'research_task','task_id':'t2','state':'waiting','phase':'main'},
                {'event':'model_request','status':'completed','role':'worker-2'},
                {'event':'research_task','task_id':'t1','state':'running','phase':'rework'}]
        state=view_state(events=events)
        self.assertEqual(state['states']['t1']['phase'],'rework')
        self.assertEqual(state['states']['t2']['state'],'waiting')
        self.assertEqual(state['phases'][('t1','main')]['state'],'completed')
        self.assertIn('等待完成：取数',research_html(events=events))
        events.append({'event':'research_task','task_id':'t2','state':'blocked','phase':'main'})
        self.assertIn('前置失败，已阻塞',research_html(events=events))

    def test_shared_page_and_no_permissions_inferred(self):
        events=[{'event':'research_task','task_id':'t1','state':'running'},
                {'event':'read_page','status':'completed','role':'worker-1','page_id':'d:1','page_number':1},
                {'event':'read_page','status':'already_read','role':'worker-2','page_id':'d:1','page_number':1}]
        state=view_state(events=events)
        self.assertEqual(len(state['pages']),1)
        self.assertEqual(state['pages'][0]['received_by'],['worker-1','worker-2'])
        self.assertNotIn('analysis_versions',state['pages'][0])
        rendered=research_html(events=events)
        self.assertIn('不代表取证模型随后已看到页面图像',rendered)
        self.assertIn('等待回答',rendered)

    def test_failed_empty_draft_is_not_completed_or_verified(self):
        result=saved_result();r=result['research']
        result.update(completed=False,termination_reason='verification_invalid_output')
        r.update(drafts=[{'version':0,'numbered_claims':[],'answer':{'unresolved':['No evidence']}}],
                 verifications=[],failure_stage='verify_v0',verification_status='incomplete')
        value=research_html(result)
        self.assertIn('任务未完成',value)
        self.assertIn('verify_v0',value)
        self.assertIn('1版草稿 / 0次完成核验',value)
        self.assertIn('No evidence',value)
        self.assertNotIn('模型核验完成',value)

    def test_old_record_without_version_is_not_matched_by_list_position(self):
        result=saved_result(); r=result['research']
        r['verifications']=[{'claims':[],'issues':[]}]
        value=research_html(result)
        self.assertIn('版本未记录',value)
        self.assertIn('草稿 v0 · 核验未完成或版本未对应',value)
        self.assertIn('草稿 v1 · 核验未完成或版本未对应',value)
        self.assertEqual(research_html(events=[{'event':'search_pages','status':'completed'}]),'')

    def test_calculations_keep_reported_and_computed_values_separate(self):
        result=saved_result()
        result['research']['drafts'][0]['calculations']=[{'operation':'divide','a':51,'b':66,
            'result':0.7727,'computed_result':51/66,'arithmetic_ok':True,'unit':'ratio'}]
        value=research_html(result)
        self.assertIn('草稿值 0.7727',value)
        self.assertIn('复算值 0.772727',value)
        self.assertIn('操作数、分母和单位仍需查看原页',value)


if __name__ == '__main__':
    unittest.main()
