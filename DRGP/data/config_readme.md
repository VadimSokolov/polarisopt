A configuration ```json``` file is used to define the POLARIS variables being calibrated and their attributes. The format is a 2-level json file with the following information:

```Filename``` - the name of the POLARIS file where a subset of variables reside. This file is assumed to be located in the same folder as the simulation executable

```Attributes``` - each variable must be defined with:
* ```name``` - the identifier of the parameter as listed in ```filename`` 
* ```type``` - designate whether the variable is of type ```float``` or ```int```
* ```min``` - the minimum value the algorithm should consider for the variable
* ```max``` - the maximum value the algorithm should consider for the variable
* ```size``` - the number of variable instances to calibrate over


---------------------------------------------------------
---------------------------------------------------------
                Example config.json setup
---------------------------------------------------------
---------------------------------------------------------
```json
{
    "BloomingtonModeChoiceModel.json": [
        {
            "name": "HBO_B_male_taxi",
            "type": "float",
            "min": "0.29800000000000004",
            "max": "2.298",
            "size": 1
        },
        {
            "name": "NHB_B_dens_bike",
            "type": "float",
            "min": "6.601",
            "max": "8.600999999999999",
            "size": 1
        },
        {
            "name": "HBO_ASC_TAXI",
            "type": "float",
            "min": "2.34",
            "max": "4.34",
            "size": 1
        }
    ],
    "BloomingtonDestinationChoiceModel.json": [
        {
            "name": "THETAR_WORK",
            "type": "float",
            "min": "-8.553",
            "max": "-6.553",
            "size": 1
        },
        {
            "name": "GAMMA_SERVICE",
            "type": "float",
            "min": "7.038",
            "max": "9.038",
            "size": 1
        }     
    ]
}

```