"""
A bunch of lists of dictionaries of destinations at various spots tested.
Possible keys:
    obstacles (not currently used)
    destType: (point/beginRow/endRow/sample)
    coord: (lat&long)
    finalHading: (degrees) - the robot must face this heading before moving to the next destination
    destTolerance (m) distance from the actual destination the robot can be before it considers it at the destination
    rowDirection (degrees) - orientation of the row. Very helpful for the robot entering the row

"""

startRow = (40.4718345, -86.9952429)
rows = []
i = 0
while i < 10:
    rows += [[[startRow[0]-i*0.00002, startRow[1]], [startRow[0]-i*0.00002, startRow[1]-0.0007]]]
    i+=1


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

acreBayCorn = [{"obstacles": []},
                    {"coord": [40.469895, -86.995335], "destType": "point", "destTolerance": 1.5},
                    {"coord": [40.469955, -86.995278], "destType": "point", "destTolerance": 1.5},

                    {"coord": [40.4699390, -86.9953355], "finalHeading": 270, "destType": "point", "destTolerance": 0.4},
                    {"coord": [40.4699390, -86.99533555], "rowShape": [[0,0],[0,1000], [1000,1000],[1000,0]], "destType": "row", "rowDirection": 270}]
                    # {"coord": [40.469810, -86.996626], "destType": "endRow"}]

acreBayCornNorth = [{"obstacles":[]},                 #   {"coord": [40.4705652, -86.9952982], "destType": "point"}, # test point inside of obstacle
                    {"coord": [40.4705473, -86.9952983], "finalHeading": 360, "destType": "point", "destTolerance": 0.4},
                    {"coord": [40.4705473, -86.9952983], "rowShape": [[0,0],[0,1000], [1000,1000],[1000,0]], "destType": "row", "rowDirection": 360},
                    {"coord": [40.470558, -86.995371], "destType": "point"}]

acreBayCornSouth = [{'obstacles': []},
                    {"coord": [40.4698295, -86.9954764], "finalHeading": 270, "destType": "point", "destTolerance": 0.4},
                    {"coord": [40.4698295, -86.9954764], "destType": "row", "rowDirection": 270},
                    {"coord": [40.4698172, -86.9968853], "destType": "point"}]

acreBayCornFarNorth = [{'obstacles': [], 'rows': rows},
                     # {"coord": [40.4717164, -86.9951846], "destType": "point", "destTolerance": 1.5},
                     {"coord": [40.4717972, -86.9951975], "destType": "point", "finalHeading": 270, "destTolerance": 0.4},
                     {"coord": [40.4717972, -86.9951975], "destType": "row", "rowDirection": 270, "rowShape": [[40.4716242, -86.9952919], [40.4718456, -86.9952928], [40.4718402,-86.9959584], [40.4716069,-86.9960634]], "destTolerance": 1.5},


                     {"coord": [40.4717703, -86.9960504], "destType": "point", "destTolerance": 1.5},
                     {"coord": [40.4718123, -86.9960236], "destType": "point", "finalHeading": 85, "destTolerance": 0.4}]
                     # {"coord": [40.4718123, -86.9960236], "destType": "row", "rowShape": [[40.4716242, -86.9952919], [40.4718456, -86.9952928], [40.4718402,-86.9959584], [40.4716069,-86.9960634]], "destTolerance": 1.5}]

endurancePath = [{'obstacles': [], 'rows': []}]


i=0
while i < 20:
    endurancePath += [{"coord": [40.4705285, -86.9953610], "destType": "point", "destTolerance": 1.5}]
    endurancePath += [{"coord": [40.4705285, -86.9963610], "destType": "point", "destTolerance": 1.5}]

    i+=1

# i=0
# while i < 5:
#     endurancePath += [{"coord": [40.4705285, -86.9953610+i*0.00002], "destType": "point", "finalHeading": 1, "destTolerance": 1.5}]
#     endurancePath += [{"coord": [40.4705285, -86.9953610+i*0.00002+0.00001], "destType": "row", "rowDirection": 1, "rowShape": [[40.4705721, -86.9952970], [40.4705651, -86.9952839], [40.4716223, -86.9953837], [40.4716177, -86.9952695]], "destTolerance": 1.5}]

#     endurancePath += [{"coord": [40.4716615, -86.9953793+i*0.00002+0.00001], "destType": "point", "finalHeading": 180, "destTolerance": 1.5}]
#     endurancePath += [{"coord": [40.4716615, -86.9953793+i*0.00002+0.00002], "destType": "row", "rowDirection": 180, "rowShape": [[40.4705721, -86.9952970], [40.4705651, -86.9952839], [40.4716223, -86.9953837], [40.4716177, -86.9952695]], "destTolerance": 1.5}]
#     i+=1



# endurancePathSoy = [{'obstacles': [], 'rows': []}]

# i=0
# while i < 10:
#     endurancePathSoy += [{"coord": [40.4693128, -86.9963120+i*0.00002], "destType": "point"}]
#     endurancePathSoy += [{"coord": [40.4693128, -86.9963120+i*0.00002+0.00001], "destType": "point", "finalHeading": 180, "destTolerance": 1.5}]
    
# # 40.4686842
#     endurancePathSoy += [{"coord": [40.4691352, -86.9963120+i*0.00002+0.00001], "destType": "point"}]
#     endurancePathSoy += [{"coord": [40.4691352, -86.9963120+i*0.00002+0.00002], "destType": "point", "finalHeading": 1, "destTolerance": 1.5}]
#     i+=1