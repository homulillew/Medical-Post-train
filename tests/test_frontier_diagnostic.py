import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_frontier import classify_group, paired_bootstrap
from analyze_stage4_postformal import first_crossing


def group(answers):
    rows=[]
    for i,a in enumerate(answers):
        rows.append(dict(prompt_id='p',group_id='g',policy_version='policy',config_sha256='config',reward_version='reward',
            trajectory_id=str(i),raw_output=('nonsense' if a is None else '<answer>'+a+'</answer>'),
            options={'A':'a','B':'b'},ground_truth='A',finish_reason='stop',output_tokens=10))
    return dict(prompt_id='p',responses=rows)


class FrontierTests(unittest.TestCase):
    def test_classes(self):
        for answers,label in [(['B']*4,'all_wrong'),(['A']*4,'all_correct'),(['A','A','B','B'],'mixed_parsed_wrong'),
                (['A','A',None,None],'mixed_unparseable_only'),(['A','B',None,None],'mixed_both')]:
            self.assertEqual(classify_group(group(answers),'policy')['classification'],label)

    def test_invalid_group_and_policy_rejected(self):
        for g,p in [(group(['A']*3),'policy'),(group(['A']*4),'other')]:
            with self.assertRaises(AssertionError):classify_group(g,p)

    def test_majority_tie_and_unparseable(self):
        self.assertEqual(classify_group(group(['A','A','B','B']),'policy')['majority_accuracy'],0)
        self.assertEqual(classify_group(group([None]*4),'policy')['consistency'],0)

    def test_paired_bootstrap(self):
        self.assertEqual(paired_bootstrap([0,0],[1,1],resamples=500)['ci95'],[1.,1.])
        self.assertEqual(paired_bootstrap([0,1],[0,1],resamples=500)['ci95'],[0.,0.])
        self.assertEqual(paired_bootstrap([0,1],[1,0],resamples=500),paired_bootstrap([0,1],[1,0],resamples=500))
        self.assertIsNone(paired_bootstrap([],[])['ci95'])

    def test_first_crossing_no_interpolation_or_future_substitution(self):
        rows=[dict(rollout_tokens=900),dict(rollout_tokens=1001),dict(rollout_tokens=2000)]
        self.assertIs(first_crossing(rows,1000),rows[1])
        self.assertIs(first_crossing(rows,2000),rows[2])
        self.assertIsNone(first_crossing(rows,2001))


if __name__=='__main__':unittest.main()
