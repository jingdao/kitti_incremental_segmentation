
#list of all classes
classes = ['clutter', 'road', 'sidewalk', 'building', 'vehicle', 'person', 'vegetation']

class_mapping = {
  0 : 0,
  1 : 0,
  10: 4,
  11: 4,
  13: 4,
  15: 4,
  16: 4,
  18: 4,
  20: 4,
  30: 5,
  31: 4,
  32: 4,
  40: 1,
  44: 2,
  48: 2,
  49: 2,
  50: 3,
  51: 3,
  52: 3,
  60: 1,
  70: 6,
  71: 6,
  72: 2,
  80: 3,
  81: 3,
  99: 0,
  252: 4,
  253: 4,
  254: 5,
  255: 4,
  256: 4,
  257: 4,
  258: 4,
  259: 4
}

#integer ID for each class
class_to_id = {classes[i] : i for i in range(len(classes))}

class_to_color_rgb = {
	0: (200,200,200), #clutter
	1: (0,100,100), #board
	2: (255,0,0), #bookcase
	3: (255,200,200), #beam
	4: (0,0,100), #chair
	5: (0,255,255), #column
	6: (0,255,0), #door
	7: (255,0,255), #sofa
	8: (50,50,50), #table
#	9: (0,255,0), #window
#	10: (255,255,0), #ceiling
#	11: (0,0,255), #floor
#	12: (255,165,0), #wall
}

if __name__=='__main__':
    import matplotlib.pyplot as plt
    plt.figure()
#    for i in [10,11,12,6,9,2,8,7,4,5,3,1,0]:
    for i in range(len(classes)):
        c = class_to_color_rgb[i]
        c = (1.0*c[0]/255, 1.0*c[1]/255, 1.0*c[2]/255)
        plt.scatter(0,0,color=c,label=classes[i],s=200)
    plt.legend(ncol=7,prop={'size': 16})
#    plt.figure()
#    for i in [11,1,2,3,4,6,7,8,12,5,9,10,13,0,14,15,16]:
#        c = class_to_color_rgb_outdoor[i]
#        c = (1.0*c[0]/255, 1.0*c[1]/255, 1.0*c[2]/255)
#        plt.scatter(0,0,color=c,label=classes_outdoor[i],s=200)
#    plt.legend(ncol=7,prop={'size': 16})
    plt.show()
