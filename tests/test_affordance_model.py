import json
import tempfile
from pathlib import Path
import unittest
import numpy as np
from autonomy.affordance_model import OutcomePredictor,features,evaluate

class AffordanceNetworkTests(unittest.TestCase):
    def test_sensor_only_inputs(self):
        x=features(.17,.56,.02,True)
        self.assertEqual(x.shape,(5,))
        self.assertTrue(np.isfinite(x).all())
        with self.assertRaises(ValueError): features(float('nan'),.5,.1,True)
    def test_learning_informative_probe(self):
        r=np.random.default_rng(3)
        values=r.uniform(0,.15,160)
        xs=np.array([features(r.uniform(.1,.4),.55,z,True) for z in values])
        labels=(values>.065).astype(float)
        model=OutcomePredictor.fresh(19)
        before=evaluate(model,xs,labels)
        model.fit(xs,labels,epochs=120)
        after=evaluate(model,xs,labels)
        self.assertLess(after['neural_brier'],before['neural_brier']-.06)
        self.assertGreater(after['neural_accuracy'],.82)
    def test_roundtrip_and_reject_bad_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'model.json'
            model=OutcomePredictor.fresh();model.save(path)
            x=features(.25,.58,.08,True)
            self.assertAlmostEqual(model.predict(x),OutcomePredictor.load(path).predict(x))
            data=json.loads(path.read_text());data['weights']['w1'][0][0]=float('nan')
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError): OutcomePredictor.load(path)
    def test_evaluation_honest_baseline(self):
        m=OutcomePredictor.fresh()
        xs=np.array([features(.20,.55,.0,True),features(.28,.56,.12,True)])
        r=evaluate(m,xs,np.array([0,1]))
        self.assertEqual(r['baseline_brier'],.25)

if __name__=='__main__':unittest.main()
