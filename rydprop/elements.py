from .atom import RydbergAtom
from .state import default_basis

def TripletHelium(basis = default_basis):
    mass = 4.002602
    defects = dict({
0 : {
            1 : [0.29665648771, 0.038296666, 0.0075131, -0.0045476, 0.002180]
    },
1 : {
            0 : [0.06832800251, -0.018641975, -0.01233165, -0.0079515, -0.005448],
            1 : [0.06835785765, -0.018630462, -0.01233040, -0.0079512, -0.005450],
            2 : [0.06836028379, -0.018629228, -0.01233275, -0.0079527, -0.005451]
    },
2 : {
            1 : [0.002885580281, -0.0063576012, 0.00033667, 0.0008394, 0.0003798],
            2 : [0.002890941493, -0.0063571836, 0.00033777, 0.0008392, 0.0004323],
            3 : [0.002891328825, -0.0063577040, 0.00033670, 0.0008395, 0.0003811]
    },
3 : {
            2 : [0.00044486989, -0.001739275, 0.00010476, 0.0000337],
            3 : [0.00044859483, -0.001727232, 0.0001524, -0.0002486],
            4 : [0.00044737927, -0.001739217, 0.00010478, 0.0000331]
    },
4 : {
            3 : [0.00012570743, -0.000796498, -0.00000980, 0.000019],
            4 : [0.00012871316, -0.000796246, -0.00001189, -0.0000141],
            5 : [0.00012714167, -0.000796484, -0.00000985, -0.000019]
    },
5 : {
            4 : [0.000047797067, -0.0004332322, -0.00000807],
            5 : [0.000049757614, -0.0004332274, -0.00000813],
            6 : [0.000048729846, -0.0004332281, -0.00000810]
    },
6 : {
            5 :[0.000022390759, -0.0002610680, -0.000004042],
            6 :[0.000023768483, -0.0002610662, -0.000004076],
            7 :[0.000023047609, -0.0002610672, -0.00000404]
    }

    })

    additional_states = dict({
        1:{
            0:-24.58738880
          },
        2:{
            0:-4.767774406320104,
            1:-3.6233020674810597,
        },
        3:{
            0:-1.8689225715263191,
            1: -1.580315853191614,
            2: -1.5137382920594895,
           },
         4:{
         0:-0.9934302978354488,
         1:-0.8794977852423038,
         2:-0.8512988244751547,
         3:-0.8503813216706249
         }
})


    return RydbergAtom(mass = mass,defects= defects,additional_states = additional_states,basis = basis)
