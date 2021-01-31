# test_capitalize.py


case 1:
 import works
case 2:
  creating the manager works
case 3:
  creating a training file or results file works by generating some samples
case 4:
imorting the samples works
case 5: 
evaluating 2+ of the samples works
case 6:
calculating the results works
case 7:
saving the results works
Case 8:
running the DR and mean work for 'non' and 'none'
case 9:
running the DR and mean work for 'NN' and 'none'
Case 10:
running the DR and mean work for 'non' and 'yes'
case 11:
running the DR and mean work for 'NN' and 'yes'
case 12:
running 2+ loop of BO works wiht case 8,9,10,11


import pytest

def test_capital_case():
    assert capital_case('semaphore') == 'Semaphore'

def test_raises_exception_on_non_string_arguments():
    with pytest.raises(TypeError):
        capital_case(9)


def capital_case(x):
    if not isinstance(x, str):
        raise TypeError('Please provide a string argument')
      return x.capitalize()


import pytest

@pytest.fixture
def example_people_data():
    return [
        {
            "given_name": "Alfonsa",
            "family_name": "Ruiz",
            "title": "Senior Software Engineer",
        },
        {
            "given_name": "Sayid",
            "family_name": "Khan",
            "title": "Project Manager",
        },
    ]

def test_format_data_for_display(example_people_data):
    assert format_data_for_display(example_people_data) == [
        "Alfonsa Ruiz: Senior Software Engineer",
        "Sayid Khan: Project Manager",
    ]

def test_format_data_for_excel(example_people_data):
    assert format_data_for_excel(example_people_data) == """given,family,title
Alfonsa,Ruiz,Senior Software Engineer
Sayid,Khan,Project Manager
"""

from PolarisOpt.setup_manager import SetupManager
import os, sys
settings_filename = "settings.json"
config_filename = "config_timedep.json"
manager = SetupManager(settings_filename, config_filename)

from PolarisOpt.F import build_sampleset
build_sampleset(manager, manager.training_filename, max_parallel = 32, num_samples=16)


or
mm
import pytest

def test_config_file(config_filename):



def test_sum():
    assert sum([1, 2, 3]) == 6, "Should be 6"

def test_sum_tuple():
    assert sum((1, 2, 2)) == 6, "Should be 6"

1973479