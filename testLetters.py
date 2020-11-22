################################################################################

# Alpacas & Fences - testLetters
# Authors: 470354850, 470386390, 470203101

# In order to run this file alone:
# $ python testLetters.py

# This script aids in the collection of web camera images.

# Code modified from https://subscription.packtpub.com/book/application_development/9781785283932/3/ch03lvl1sec28/accessing-the-webcam

################################################################################
# Imports
################################################################################
import cv2
import numpy as np
import matplotlib.pyplot as plt
import idx2numpy

from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split

from skimage.feature import hog
import pickle

################################################################################
# Main
################################################################################
if __name__ == "__main__":

    #filename = 'c:/Users/strat/OneDrive/Desktop/majorProjectCV/finalised_model.sav'
    filename = 'c:/Users/strat/OneDrive/Desktop/majorProjectCV/ROGM_model.sav'
    
    loaded_model = pickle.load(open(filename, 'rb'))

    # Starting off with letters R O Y G C B P M
    letters_To_Track = [18, 15, 25, 7, 3, 2, 16, 13]

    featureList = []
    plot_Number = 1

    #testSet1 = {0: [(77, 82), (76, 83), (76, 84), (77, 84), (75, 86), (75, 86), (74, 85), (73, 85), (74, 84), (75, 83), (74, 88), (73, 103), (74, 128), (76, 141), (77, 156), (78, 172), (77, 188), (79, 218), (75, 248), (73, 263), (74, 277), (73, 290), (71, 302), (71, 313), (74, 323), (73, 341), (74, 350), (72, 350), (70, 347), (70, 341), (69, 333), (68, 322), (62, 295), (59, 280), (56, 264), (55, 247), (52, 216), (58, 183), (63, 152), (75, 121), (82, 107), (89, 95), (99, 83), (111, 73), (133, 60), (146, 55), (161, 51), (187, 49), (217, 50), (247, 57), (259, 62), (274, 68), (283, 76), (295, 83), (306, 90), (313, 98), (319, 106), (325, 112), (327, 117), (332, 120), (332, 122), (332, 122), (332, 121), (332, 120), (330, 119), (326, 118), (324, 118), (327, 117)], 1: [(303, 138), (304, 139), (303, 140), (308, 139), (306, 139), (307, 139), (311, 138), (310, 137), (313, 136), (311, 136), (309, 136), (308, 135), (302, 134), (281, 132), (268, 130), (251, 129), (230, 130), (194, 135), (177, 141), (160, 149), (144, 158), (132, 168), (102, 194), (87, 221), (78, 236), (76, 249), (72, 262), (73, 274), (72, 288), (75, 301), (91, 325), (102, 336), (115, 346), (125, 355), (159, 367), (187, 373), (202, 373), (217, 371), (243, 360), (266, 341), (278, 316), (291, 286), (298, 254), (299, 222), (296, 207), (292, 193), (289, 180), (283, 168), (263, 148), (253, 140), (240, 133), (225, 127), (209, 123), (198, 119), (183, 116), (160, 110), (151, 107), (138, 105), (134, 102), (125, 99), (126, 97), (127, 96), (126, 96), (128, 96), (131, 97), (129, 98)], 2: [(101, 87), (102, 87), (100, 87), (99, 87), (100, 87), (101, 87), (96, 88), (93, 89), (92, 90), (86, 93), (82, 97), (80, 110), (76, 130), (75, 142), (77, 168), (79, 182), (82, 196), (93, 207), (173, 223), (187, 231), (198, 232), (207, 232), (219, 230), (192, 222), (210, 206), (222, 195), (226, 184), (238, 159), (237, 146), (238, 133), (242, 120), (242, 98), (242, 88), (240, 79), (239, 70), (239, 61), (242, 52), (245, 44), (246, 37), (250, 31), (248, 28), (249, 29), (250, 41), (254, 61), (254, 75), (259, 90), (263, 105), (269, 137), (272, 153), (273, 169), (276, 185), (283, 201), (288, 218), (291, 235), (312, 248), (306, 262), (311, 274), (302, 286), (298, 296), (290, 306), (283, 314), (275, 321), (267, 327), (257, 332), (249, 335), (240, 337), (233, 338), (220, 339), (208, 339), (196, 338), (184, 336), (173, 333), (162, 329), (155, 324), (148, 319), (137, 314), (125, 309), (112, 304), (104, 298), (95, 292), (87, 285), (79, 278), (71, 271), (73, 263), (66, 256), (59, 250), (66, 244), (57, 241), (64, 238), (64, 237), (62, 238), (69, 239), (67, 242), (62, 244), (62, 247), (69, 249)], 3: [(328, 118), (327, 118), (326, 118), (331, 118), (336, 118), (341, 118), (343, 115), (344, 102), (341, 91), (330, 79), (318, 67), (303, 56), (284, 48), (264, 44), (227, 44), (208, 48), (198, 54), (190, 63), (181, 76), (170, 91), (159, 125), (165, 157), (182, 182), (209, 197), (236, 205), (251, 205), (275, 197), (294, 181), (301, 170), (309, 157), (313, 144), (314, 118), (309, 106), (304, 95), (296, 75), (286, 67), (279, 52), (275, 46), (274, 41), (273, 37), (267, 35), (273, 36), (267, 41), (272, 47), (273, 56), (273, 66), (277, 76), (283, 99), (292, 125), (291, 141), (301, 174), (299, 206), (297, 221), (292, 248), (283, 262), (275, 275), (261, 287), (253, 295), (235, 302), (215, 308), (201, 312), (215, 316), (205, 316), (187, 316), (179, 313), (162, 310), (155, 304), (149, 298), (146, 292), (73, 284), (70, 282), (73, 280)], 4: [(315, 105), (319, 104), (319, 104), (322, 103), (326, 102), (327, 102), (331, 101), (329, 101), (338, 99), (344, 97), (349, 92), (346, 88), (345, 82), (341, 75), (334, 67), (325, 58), (314, 50), (304, 42), (289, 36), (276, 30), (259, 27), (244, 25), (210, 28), (193, 32), (176, 38), (166, 45), (149, 55), (126, 80), (116, 95), (109, 111), (106, 127), (103, 144), (103, 160), (101, 191), (107, 204), (116, 229), (121, 241), (130, 252), (145, 260), (167, 272), (180, 274), (210, 272), (223, 269), (233, 265), (242, 260), (256, 253), (272, 240), (283, 232), (288, 225), (294, 218), (297, 212), (306, 205), (306, 200), (309, 191), (310, 188), (312, 186), (308, 187), (306, 190)], 5: [(111, 68), (104, 74), (107, 78), (110, 80), (106, 82), (104, 83), (107, 82), (106, 81), (109, 80), (106, 84), (103, 102), (105, 117), (103, 136), (104, 157), (110, 179), (114, 203), (111, 229), (114, 254), (122, 278), (116, 302), (118, 345), (118, 362), (122, 382), (125, 384), (124, 382), (121, 375), (116, 364), (114, 350), (110, 334), (109, 316), (107, 297), (107, 278), (110, 259), (120, 239), (124, 221), (130, 205), (142, 190), (151, 178), (169, 168), (180, 163), (196, 161), (214, 162), (231, 166), (244, 173), (256, 181), (266, 190), (273, 200), (274, 212), (275, 225), (272, 239), (271, 253), (266, 266), (262, 277), (250, 288), (235, 298), (219, 306), (201, 313), (179, 320), (159, 326), (138, 331), (120, 335), (98, 338), (80, 339), (61, 339), (43, 334), (32, 332), (30, 329), (24, 326), (24, 325), (26, 324), (25, 324), (29, 325), (31, 327), (29, 328), (29, 328)], 6: [(140, 72), (144, 72), (140, 73), (140, 74), (141, 75), (136, 78), (137, 78), (133, 79), (134, 79), (130, 81), (133, 83), (130, 88), (124, 111), (123, 129), (123, 150), (126, 195), (121, 218), (124, 239), (123, 277), (123, 294), (124, 309), (126, 322), (127, 333), (130, 341), (127, 349), (125, 349), (120, 346), (114, 330), (105, 303), (107, 268), (102, 230), (100, 211), (108, 190), (110, 171), (130, 135), (141, 119), (164, 96), (203, 85), (235, 88), (254, 92), (286, 105), (299, 114), (308, 125), (311, 138), (312, 151), (308, 164), (301, 176), (277, 199), (264, 209), (246, 218), (227, 225), (204, 230), (183, 233), (159, 234), (132, 234), (93, 227), (76, 221), (56, 206), (47, 193), (49, 188), (44, 189), (45, 190), (49, 191), (50, 192), (51, 193), (47, 195)], 7: [(54, 99), (54, 99), (55, 98), (54, 97), (53, 96), (52, 95), (54, 93), (56, 91), (56, 90), (54, 90), (52, 91), (56, 98), (53, 105), (54, 113), (58, 122), (63, 144), (68, 157), (60, 174), (59, 192), (59, 210), (57, 228), (56, 246), (58, 263), (55, 278), (56, 290), (57, 299), (57, 304), (58, 304), (58, 299), (62, 290), (62, 279), (63, 250), (60, 233), (66, 213), (71, 193), (79, 173), (90, 153), (104, 135), (117, 121), (130, 111), (139, 106), (152, 106), (163, 112), (173, 122), (177, 137), (179, 155), (184, 174), (181, 215), (179, 236), (183, 252), (186, 263), (186, 270), (187, 273), (190, 271), (192, 266), (195, 260), (195, 254), (191, 248), (189, 239), (188, 227), (189, 211), (187, 172), (206, 126), (220, 106), (233, 89), (247, 77), (264, 70), (276, 70), (293, 74), (314, 100), (320, 118), (322, 140), (322, 163), (321, 186), (321, 208), (317, 229), (318, 247), (320, 263), (321, 277), (319, 288), (320, 296), (323, 301), (329, 303), (329, 304), (333, 304), (327, 304), (327, 302), (324, 300), (324, 297), (324, 295), (327, 293)]}
    #testSet2 = {0: [(101, 84), (99, 85), (100, 86), (99, 86), (100, 85), (100, 86), (101, 86), (104, 84), (103, 84), (102, 85), (105, 86), (98, 92), (96, 99), (94, 109), (91, 123), (86, 156), (86, 174), (84, 192), (83, 211), (90, 245), (88, 261), (87, 277), (89, 301), (89, 311), (89, 319), (84, 315), (84, 295), (79, 281), (82, 263), (82, 244), (95, 201), (101, 180), (125, 140), (140, 122), (171, 95), (207, 79), (227, 76), (245, 76), (264, 78), (280, 82), (289, 87), (298, 91), (301, 94), (304, 95), (304, 95), (303, 94), (301, 92), (290, 91), (286, 89), (282, 90), (282, 90)], 1: [(200, 79), (199, 78), (198, 78), (196, 76), (192, 76), (192, 75), (189, 73), (177, 74), (162, 77), (149, 83), (117, 108), (103, 126), (81, 170), (79, 193), (76, 217), (77, 239), (78, 261), (99, 297), (110, 311), (144, 325), (162, 325), (179, 322), (204, 313), (224, 301), (255, 274), (270, 257), (290, 220), (296, 200), (296, 180), (294, 160), (290, 139), (281, 121), (250, 91), (226, 81), (199, 75), (172, 72), (129, 73), (102, 79), (95, 82), (92, 84), (88, 85), (88, 85), (93, 84), (90, 84), (89, 84), (90, 84)], 2: [(107, 49), (104, 50), (103, 51), (104, 50), (103, 49), (104, 49), (101, 50), (105, 49), (103, 49), (101, 52), (100, 53), (99, 54), (101, 56), (99, 58), (96, 61), (95, 66), (94, 74), (93, 84), (90, 97), (90, 113), (91, 130), (94, 146), (100, 161), (112, 174), (122, 185), (133, 193), (147, 198), (161, 201), (177, 201), (188, 200), (200, 197), (214, 191), (222, 183), (236, 171), (242, 159), (251, 145), (255, 130), (261, 114), (262, 98), (263, 83), (263, 68), (266, 54), (262, 44), (260, 36), (260, 30), (259, 27), (259, 28), (262, 32), (261, 40), (265, 50), (264, 64), (266, 79), (266, 95), (265, 113), (267, 131), (266, 151), (267, 172), (271, 193), (263, 217), (261, 238), (250, 259), (242, 277), (229, 293), (216, 306), (204, 316), (184, 325), (167, 330), (148, 332), (132, 331), (112, 328), (99, 321), (80, 301), (69, 280), (62, 266), (59, 263), (58, 261), (58, 261), (59, 261), (59, 261), (63, 260), (62, 260), (60, 260), (60, 259), (58, 259)], 3: [(253, 133), (250, 134), (250, 135), (252, 136), (250, 137), (251, 137), (252, 138), (254, 137), (256, 132), (252, 120), (232, 107), (217, 102), (201, 98), (183, 97), (166, 99), (152, 104), (135, 113), (109, 144), (101, 182), (97, 203), (95, 241), (106, 272), (116, 284), (129, 293), (140, 300), (155, 304), (168, 305), (184, 302), (199, 297), (212, 290), (226, 280), (247, 259), (256, 249), (262, 240), (265, 233), (274, 224), (274, 222), (271, 222), (273, 222), (271, 223), (275, 223), (276, 224)], 4: [(132, 95), (129, 80), (131, 74), (129, 69), (131, 65), (129, 63), (128, 61), (126, 57), (126, 55), (123, 50), (119, 46), (119, 44), (118, 48), (117, 55), (118, 64), (119, 92), (120, 108), (121, 126), (118, 147), (117, 168), (119, 187), (114, 225), (115, 242), (112, 259), (109, 275), (105, 290), (103, 301), (102, 308), (103, 312), (103, 314), (105, 310), (101, 304), (102, 294), (101, 281), (102, 265), (101, 248), (105, 230), (113, 211), (118, 193), (126, 176), (136, 161), (152, 147), (164, 137), (182, 130), (202, 127), (223, 127), (244, 131), (266, 138), (283, 149), (302, 161), (314, 176), (316, 194), (324, 211), (321, 230), (321, 249), (319, 268), (307, 287), (294, 304), (276, 318), (253, 330), (231, 339), (202, 346), (171, 350), (146, 349), (122, 345), (108, 337), (91, 328), (81, 317), (78, 305), (72, 295), (66, 283), (66, 280), (64, 280), (67, 280), (67, 281), (67, 282), (69, 282), (69, 282), (68, 281), (70, 280)], 5: [(105, 61), (106, 61), (108, 60), (110, 60), (108, 61), (109, 61), (111, 62), (111, 69), (110, 75), (112, 84), (113, 96), (117, 110), (112, 151), (106, 196), (102, 219), (93, 267), (94, 313), (92, 335), (93, 352), (93, 364), (96, 369), (100, 368), (99, 362), (99, 351), (97, 336), (91, 319), (88, 299), (85, 278), (82, 257), (81, 235), (82, 213), (90, 190), (95, 169), (104, 149), (114, 130), (126, 114), (143, 100), (164, 88), (181, 82), (198, 81), (214, 85), (231, 92), (242, 103), (246, 116), (244, 131), (238, 146), (231, 160), (215, 175), (197, 189), (156, 210), (137, 216), (114, 219), (90, 219), (56, 209), (44, 201), (31, 185), (27, 180), (27, 175), (31, 176), (31, 177)], 6: [(76, 141), (77, 145), (77, 150), (77, 156), (78, 161), (76, 167), (77, 174), (79, 183), (77, 194), (79, 226), (77, 246), (78, 285), (73, 302), (72, 313), (74, 317), (71, 317), (72, 299), (79, 264), (79, 246), (79, 228), (88, 208), (104, 175), (117, 162), (134, 153), (150, 151), (162, 156), (178, 165), (186, 181), (194, 200), (205, 245), (208, 267), (208, 287), (208, 307), (208, 306), (207, 298), (212, 281), (204, 261), (202, 238), (202, 214), (203, 191), (206, 170), (214, 151), (229, 135), (245, 124), (263, 119), (280, 121), (297, 129), (312, 142), (341, 181), (350, 204), (358, 226), (366, 262), (361, 274), (355, 288), (352, 292), (347, 295), (341, 297), (335, 296), (332, 295), (337, 293), (334, 293), (334, 293), (335, 293), (335, 293), (334, 292), (330, 281), (329, 275), (327, 275), (332, 275), (335, 275), (335, 276)]}
    
    ts1 = {0: [(71, 95), (73, 95), (77, 93), (80, 92), (84, 89), (89, 86), (87, 84), (86, 83), (88, 82), (84, 84), (84, 89), (82, 98), (83, 107), (83, 131), (81, 145), (81, 158), (83, 172), (80, 186), (80, 199), (81, 212), (81, 225), (83, 238), (81, 252), (83, 264), (86, 275), (87, 286), (89, 295), (91, 311), (97, 319), (95, 333), (92, 339), (95, 339), (91, 335), (94, 327), (90, 317), (90, 305), (87, 293), (85, 266), (84, 238), (89, 224), (87, 212), (90, 199), (95, 186), (98, 175), (104, 163), (124, 140), (132, 131), (140, 123), (152, 117), (164, 113), (178, 110), (208, 109), (219, 111), (234, 112), (248, 114), (277, 121), (303, 129), (309, 133), (316, 135), (316, 137), (315, 137), (313, 135), (310, 135), (309, 135)], 1: [(233, 89), (234, 89), (232, 89), (230, 89), (230, 88), (226, 86), (225, 84), (213, 82), (203, 83), (193, 86), (183, 92), (155, 117), (140, 135), (126, 157), (108, 206), (101, 232), (100, 284), (113, 328), (141, 353), (158, 361), (177, 365), (199, 365), (220, 362), (255, 347), (266, 336), (281, 322), (293, 307), (297, 292), (306, 259), (307, 241), (308, 223), (309, 188), (301, 173), (285, 144), (276, 131), (251, 110), (233, 102), (215, 95), (197, 90), (182, 86), (146, 87), (133, 90), (119, 95), (112, 98), (109, 100), (106, 102), (108, 102), (108, 102), (108, 102), (110, 102)], 2: [(68, 92), (70, 92), (69, 93), (73, 93), (72, 94), (74, 94), (76, 94), (76, 94), (76, 94), (78, 93), (79, 92), (79, 92), (82, 92), (84, 94), (93, 108), (93, 122), (94, 139), (100, 158), (105, 179), (104, 226), (106, 250), (105, 274), (106, 296), (105, 315), (101, 330), (101, 337), (95, 338), (90, 333), (89, 320), (86, 303), (85, 282), (87, 257), (82, 232), (79, 207), (87, 158), (93, 137), (113, 101), (127, 87), (140, 78), (153, 75), (166, 78), (178, 86), (189, 99), (195, 117), (203, 138), (208, 163), (216, 188), (220, 213), (223, 257), (224, 272), (217, 289), (217, 289), (208, 278), (205, 265), (209, 247), (207, 227), (201, 206), (200, 159), (199, 137), (208, 96), (217, 79), (228, 65), (237, 57), (264, 55), (277, 60), (292, 68), (305, 78), (325, 104), (334, 119), (344, 155), (352, 173), (355, 192), (354, 211), (357, 227), (356, 241), (360, 252), (363, 267), (363, 271), (363, 276), (365, 277), (363, 279), (363, 280), (362, 281), (360, 283), (360, 284), (361, 285), (358, 286), (360, 286)], 3: [(336, 101), (335, 104), (338, 105), (340, 108), (341, 109), (342, 110), (343, 113), (347, 116), (352, 115), (354, 111), (353, 103), (347, 92), (332, 81), (307, 74), (273, 72), (234, 74), (200, 81), (171, 94), (146, 112), (127, 135), (120, 159), (119, 184), (131, 205), (152, 222), (172, 234), (200, 240), (225, 240), (276, 223), (293, 209), (299, 192), (313, 171), (315, 150), (311, 106), (308, 85), (306, 66), (303, 51), (302, 41), (299, 39), (302, 43), (309, 52), (317, 65), (322, 81), (327, 98), (333, 119), (338, 142), (339, 167), (344, 191), (360, 235), (354, 256), (350, 275), (333, 306), (316, 319), (307, 328), (272, 342), (256, 346), (233, 347), (224, 345), (208, 341), (193, 335), (184, 327), (178, 318), (147, 308), (123, 297), (121, 286), (116, 278), (116, 271), (112, 270), (106, 270), (105, 268), (105, 267), (105, 265), (101, 265)]}
    ts2 = {0: [(71, 70), (73, 69), (72, 68), (70, 66), (68, 65), (68, 63), (66, 61), (67, 63), (66, 69), (68, 79), (65, 94), (67, 129), (67, 149), (63, 191), (66, 211), (67, 230), (67, 250), (65, 269), (65, 287), (66, 302), (60, 313), (61, 316), (61, 313), (66, 304), (70, 291), (70, 275), (74, 257), (73, 239), (83, 200), (94, 180), (107, 161), (119, 145), (134, 130), (151, 118), (171, 108), (214, 98), (237, 98), (275, 107), (287, 113), (293, 119), (300, 122), (305, 123), (306, 123), (308, 122), (314, 120), (310, 119), (308, 117), (308, 116), (305, 116), (307, 115)], 1: [(197, 86), (190, 86), (185, 86), (183, 86), (176, 88), (175, 89), (168, 91), (160, 94), (146, 101), (106, 127), (86, 148), (66, 174), (56, 203), (50, 234), (61, 264), (68, 291), (85, 315), (104, 336), (131, 351), (157, 363), (210, 368), (235, 359), (260, 342), (274, 320), (293, 294), (309, 267), (314, 239), (318, 210), (316, 182), (290, 133), (277, 112), (249, 96), (225, 83), (196, 75), (165, 71), (140, 69), (125, 68), (111, 69), (104, 71), (104, 72), (109, 72), (108, 73), (110, 73), (105, 74), (106, 74)], 2: [(83, 63), (84, 61), (82, 60), (80, 60), (81, 61), (77, 70), (76, 79), (87, 113), (81, 175), (81, 209), (81, 240), (79, 289), (77, 304), (74, 312), (75, 288), (78, 266), (74, 213), (79, 184), (81, 158), (113, 117), (130, 108), (153, 109), (178, 120), (193, 141), (203, 169), (206, 201), (218, 230), (214, 255), (213, 269), (211, 266), (210, 251), (211, 229), (205, 177), (214, 149), (212, 123), (228, 98), (246, 79), (272, 66), (301, 62), (323, 70), (351, 85), (367, 108), (379, 134), (387, 193), (382, 220), (382, 241), (378, 256), (373, 268), (371, 268), (372, 266), (369, 265), (372, 264), (367, 265), (365, 268), (366, 269), (367, 271)], 3: [(255, 69), (247, 69), (243, 70), (245, 72), (245, 73), (249, 73), (256, 72), (265, 69), (277, 54), (256, 32), (225, 28), (190, 32), (119, 66), (92, 96), (75, 131), (68, 169), (72, 203), (80, 230), (99, 245), (146, 244), (172, 228), (195, 205), (216, 179), (217, 152), (226, 125), (222, 102), (221, 82), (215, 67), (214, 57), (219, 51), (219, 53), (229, 61), (241, 75), (256, 95), (269, 121), (309, 183), (320, 217), (339, 280), (340, 307), (335, 328), (325, 344), (302, 357), (270, 367), (244, 372), (241, 374), (172, 372), (149, 364), (131, 353), (118, 340), (112, 327), (108, 315), (103, 307), (98, 302), (98, 299), (96, 298), (98, 298), (102, 298), (99, 299), (99, 300), (98, 301)]}
    
    for x in ts2:
        currentSet = ts2[x]
        image = np.zeros((432,432))

        for n in currentSet:
            #image[i[1],i[0]] = 1

            for i in range(-10,10) :
                for j in range(-10,10) :
                    if (n[1] + i) >= 0 and (n[1] + i) <= 431 and (n[0] + j) >= 0 and (n[0] + j) <= 431 :
                    
                        image[n[1]+i,n[0]+j] = 1


        #resizedIm = cv2.resize(image,(28,28))
        #kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(20,20))
        #morphedIm = cv2.dilate(image,kernel,iterations = 1)
        resizedIm = cv2.resize(image,(28,28))
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
        morphedIm = cv2.dilate(resizedIm,kernel,iterations = 1)
        kernel = np.ones((2,2),np.float32)/4
        dst = cv2.filter2D(morphedIm,-1,kernel)

        hog_Feature = hog(dst, orientations=8, pixels_per_cell=(4, 4), 
            cells_per_block=(1, 1), visualize=False, multichannel=False)

        featureList.append(hog_Feature)
        y_pred = loaded_model.predict(featureList)
            
        print(y_pred)
        
        plt.subplot(5, 5, plot_Number) 
        plot_Number += 1
        plt.imshow(dst, cmap=plt.get_cmap('gray'))
        

    plt.show()
    print("END PROGRAM")