import json
import tempfile
from pathlib import Path
import unittest
from autonomous_world import load_experience

class IntegratedReplayTests(unittest.TestCase):
    def test_training_rows_strip_simulator_truth(self):
        rows=[{'features':[.1,.2,.3,1,0],
               'label_meaningful_total_movement':1,
               'evaluation_only':{'hidden_mass_kg':999,'hidden_friction':.2}},
              {'valid':False,'reason':'no_touch'}]
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'prior.json';path.write_text(json.dumps(rows))
            result=load_experience(path)
        self.assertEqual(result,[{'features':[.1,.2,.3,1.,0.],'label':1}])
    def test_reject_invalid_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'bad.json'
            path.write_text(json.dumps([{'features':[.1]*5,'label':2}]))
            with self.assertRaises(ValueError):load_experience(path)
if __name__=='__main__':unittest.main()
