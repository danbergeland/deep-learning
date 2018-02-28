#!/usr/bin/python3

import unittest
import network_builder as nb

class TestNetworkBuilder(unittest.TestCase):
    def test_makeLSTMmodel_exists(self):
        model = nb.makeLSTMmodel(40)
        self.assertNotEqual(model,None)

if __name__ == "__main__":
    unittest.main()