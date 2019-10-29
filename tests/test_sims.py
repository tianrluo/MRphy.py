import unittest
import mrphy


class TestBlochSim(unittest.TestCase):

    def test_HelloWorld(self):
        print('in test_HelloWorld')
        self.assertEqual('hello'.upper(), 'HELLO')

    pass


if __name__ == '__main__':
    unittest.main()
