import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))

import unittest
from PolarisOpt import sampler, manager


class SamplerTesting(unittest.TestCase):
    def setUp(self):
        self.s = sampler.MorrisSampler('/home/vsokolov/projects/timing-austin/test/var_config.json','/home/vsokolov/projects/timing-austin/test/test-study',8,4)
        self.a = self.s.samples[0]
    def testsize(self):
        self.assertGreater(len(self.s.samples),0)
    def testsamplesize(self):
        self.assertEqual(len(self.a.input),4)
        self.assertEqual(self.a.status,'pending')
    def teststatus(self):
        s = self.s
        for item in s.samples: item.status = 'finished'
        self.assertEqual(len(s.getsamples(max=20)),0)
        for item in s.samples[35:]: item.status = 'pending'
        self.assertEqual(len(s.getsamples(100)),5)
        for item in s.samples: item.status = 'pending'
        self.assertEqual(len(s.getsamples(30)),30)
        self.assertEqual(len(s.getsamples(30)),10)
        self.assertEqual(len(s.getsamples(30)),0)


if __name__ == '__main__':
    unittest.main()

