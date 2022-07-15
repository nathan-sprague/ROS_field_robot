# # locations at ACRE corn field
acreLocations = [{"coord": [40.471649, -86.994065], "destType": "point"},
                    {"coord": [40.471569, -86.994081], "destType": "point"},
                    {"coord": [40.471433, -86.994069], "destType": "point"},
                    {"coord": [40.471399, -86.994084], "destType": "point"},
                    {"coord": [40.471597, -86.994088], "destType": "point"}]

# # locations at ACRE grass
acreGrass = [{"coord": [40.469552, -86.994882], "destType": "point"},
                    {"coord": [40.469521, -86.994578], "destType": "point"},
                    {"coord": [40.469386, -86.994755], "destType": "point"},
                    {"coord": [40.469506, -86.994384], "destType": "point"},
                    {"coord": [40.469257, -86.994658], "destType": "point"}]


# locations west of ABE building
abeWest = [{"coord": [40.4216702, -86.9184231], "heading": 0, "destType": "point"},
                    {"coord": [40.4215696, -86.9185767], "heading": 0, "destType": "point"},
                    {"coord": [40.4215696, -86.9185767], "heading": 0, "destType": "beginRow"},
                    {"coord": [40.4215696, -86.9185767], "destType": "sample"},
                    {"coord": [40.4215696, -86.9185767], "destType": "endRow"},
                    {"coord": [40.4217325, -86.9187132], "heading": 0, "destType": "point"}]


# locations north of ABE building
abeNorth = [{"coord": [40.422266, -86.916176], "destType": "point"},
                    {"coord": [40.422334, -86.916240], "destType": "point"},
                    {"coord": [40.422240, -86.916287], "destType": "point"},
                    {"coord": [40.422194, -86.916221], "destType": "point"},
                    {"coord": [40.422311, -86.916329], "destType": "point"}]

acreBayGrass = [{"coord": [40.470283, -86.995008], "destType": "point"},
                    {"coord": [40.470155, -86.995092], "destType": "point"},
                    {"coord": [40.470048, -86.995006], "destType": "point"},
                    {"coord": [40.469964, -86.995044], "destType": "point"},
                    {"coord": [40.470160, -86.995063], "destType": "point"}]

acreBayCorn = [{"obstacles": [ [[40.46977123481278,-86.99552120541317], [40.46993606641275,-86.99534717740178], [40.4701423474349,-86.99534608852393], [40.47013772209357,-86.99572404600923], [40.469767727757464,-86.99572404600923] ] ]},
                    {"coord": [40.469895, -86.995335], "destType": "point", "destTolerance": 1.5},
                    {"coord": [40.469955, -86.995278], "destType": "point", "destTolerance": 1.5},

                    {"coord": [40.4699485,-86.9953284], "finalHeading": 270, "destType": "point", "destTolerance": 0.4},
                    {"coord": [40.4699485,-86.9953284], "destType": "row"}]
                   # {"coord": [40.469810, -86.996626], "destType": "endRow"}]

acreBayCornNorth = [{"obstacles": [ [[40.470660, -86.995236], [40.470560, -86.995229], [40.470554, -86.995360], [40.469927, -86.995355], [40.469834, -86.995500], [40.469767, -86.995515], [40.469751, -86.996771], [40.470864, -86.996777] ] ]},
                 #   {"coord": [40.4705652, -86.9952982], "destType": "point"}, # test point inside of obstacle
                    {"coord": [40.4705552, -86.9952982], "finalHeading": 1, "destType": "point"},
                    {"coord": [40.4705552, -86.9952982], "destType": "row"},
                    {"coord": [40.470558, -86.995371], "destType": "point"},
                    {"coord": [40.470558, -86.995259], "destType": "point"}
                    ]
                    