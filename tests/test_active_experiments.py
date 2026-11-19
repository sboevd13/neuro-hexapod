"""Pre-action sensor-only active learning regression and acquisition tests."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np

from autonomy.active_experiments import (ACTION_DURATIONS, OutcomeEnsemble,
    acquire, choose_action, dataset, experiment_features, regression_metrics, sanitize)
from active_learning_benchmark import from_existing


class ActiveExperimentTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(2)
        self.x=rng.uniform(-.8,.8,(90,6))
        # Synthetic function only to verify training mechanics, NOT evidence of real physics gains.
        self.y=np.clip(.025+.06*self.x[:,2]+.015*self.x[:,0]+.015*self.x[:,5],-.06,.30)

    def test_experiment_action_in_features(self):
        sample=np.zeros(5)
        a=experiment_features(sample,ACTION_DURATIONS[0])
        b=experiment_features(sample,ACTION_DURATIONS[1])
        self.assertEqual(a.shape,(6,))
        self.assertNotEqual(a[-1],b[-1])
        with self.assertRaises(ValueError):experiment_features(sample,100)

    def test_sensor_only_replay_excludes_simulator_truth(self):
        one={'valid':True,'fell':False,'features':[0.,0.,.1,1.,0.],
             'total_sensed_displacement_m':.04,'second_push_seconds':ACTION_DURATIONS[0],
             'evaluation_only':{'hidden_mass_kg':.05,'hidden_friction':2.0,'mobile':True}}
        cleaned=sanitize([one])
        self.assertEqual(set(cleaned[0]),{'features','displacement_m','action_seconds'})
        self.assertNotIn('mass',json.dumps(cleaned))
        with self.assertRaises(ValueError):dataset(cleaned)
        x,y=dataset(cleaned+cleaned)
        self.assertEqual(x.shape,(2,6))
        self.assertEqual(y.tolist(),[.04,.04])

    def test_ensemble_learns_synthetic_outcome_better_than_mean(self):
        net=OutcomeEnsemble.fit(self.x[:60],self.y[:60],seed=11,members=3,steps=80)
        m=regression_metrics(net,self.x[60:],self.y[60:],baseline=float(self.y[:60].mean()))
        self.assertLess(m['mae_m'],m['constant_mae_m'])
        self.assertGreaterEqual(m['mean_ensemble_disagreement_m'],0.)

    def test_model_roundtrip_and_corruption_rejection(self):
        net=OutcomeEnsemble.fit(self.x[:30],self.y[:30],seed=9,members=3,steps=20)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model.json';net.save(path)
            other=OutcomeEnsemble.load(path)
            self.assertTrue(np.allclose(net.predict(self.x[:3])[0],other.predict(self.x[:3])[0]))
            payload=json.loads(path.read_text());payload['models'][0]['w1']=[[float('nan')]]
            path.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):OutcomeEnsemble.load(path)

    def test_active_query_only_needs_features_not_labels(self):
        net=OutcomeEnsemble.fit(self.x[:15],self.y[:15],seed=8,members=3,steps=10)
        for name in ('random','heuristic','uncertainty'):
            sel=acquire(name,net,self.x[:15],self.x[15:],7,np.random.default_rng(19))
            self.assertEqual(len(set(sel)),7)
            self.assertTrue(all(0<=i<75 for i in sel))
        # No future y is ever accepted in acquisition API.
        with self.assertRaises(TypeError):acquire('uncertainty',net,self.x[:15],self.x[15:],7,np.random.default_rng(19),self.y[15:])

    def test_safety_veto_and_bounded_action(self):
        net=OutcomeEnsemble.fit(self.x[:20],self.y[:20],seed=6,members=3,steps=15)
        self.assertEqual(choose_action(net,self.x[0,:5],safety_ok=False)['action'],'observe')
        decision=choose_action(net,self.x[0,:5],safety_ok=True)
        self.assertIn(decision['action'],('observe','gentle_push','long_push'))
        self.assertIn(decision['duration'],(0.,*ACTION_DURATIONS))

    def test_epistemic_probe_is_gentle_and_safety_bounded(self):
        class DisagreeingEnsemble:
            def predict(self, xs):
                return np.array([-.03,-.04]),np.array([.008,.009])
        probe=choose_action(DisagreeingEnsemble(),self.x[0,:5],novelty_weight=2.)
        self.assertEqual(probe['action'],'gentle_push')
        self.assertEqual(probe['duration'],ACTION_DURATIONS[0])
        veto=choose_action(DisagreeingEnsemble(),self.x[0,:5],safety_ok=False,novelty_weight=2.)
        self.assertEqual(veto['action'],'observe')

    def test_same_initial_labels_and_pure_label_budgets(self):
        train=[{'features':x.tolist(),'displacement_m':float(y)} for x,y in zip(self.x[:70],self.y[:70])]
        test=[{'features':x.tolist(),'displacement_m':float(y)} for x,y in zip(self.x[70:],self.y[70:])]
        report,models,selected=from_existing(train,test,[8,16],repeat_ids=(3,),members=3,steps=10)
        first=[r for r in report['individual_curves'] if r['budget']==8]
        self.assertEqual(len({r['metrics']['mae_m'] for r in first}),1)
        self.assertEqual(len(selected['uncertainty']),16)
        self.assertEqual(len(report['aggregate']),6)


if __name__=='__main__':unittest.main()
