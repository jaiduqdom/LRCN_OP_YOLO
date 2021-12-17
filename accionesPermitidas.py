# Despues de ejecutar el proceso videosPorAccion.py, hemos detectado que hay ciertas acciones de STAIR donde
# hay muy pocos videos disponibles. Estas acciones las dejamos excluidas ya que no permiten entrenar bien el modelo
# Autor: Jaime Duque Domingo
""" 
Hay ciertas acciones que son excluidas debido a que tenemos muy pocos videos. Concretamente:
doing_origami; 1
kissing; 17
playing_with_toy; 4
running_around; 74

El resto de acciones que si procesamos son:
assisting_in_getting_up; 1089
assisting_in_walking; 756
baby_crawling; 528
baby_crying; 685
being_angry; 654
being_surprised; 544
bottle-feeding_baby; 625
bowing; 1148
brushing_teeth; 1197
caressing_head; 1110
changing_baby_diaper; 738
clapping_hands; 934
crying; 619
cutting_food; 397
doing_high_five; 702
doing_paper-rock-scissors; 1189
drinking; 510
drying_hair_with_blower; 887
eating_meal; 441
eating_snack; 454
entering_room; 938
exercising; 304
feeding_baby; 804
fighting; 272
folding_laundry; 1288
gardening; 142
gargling; 968
giving_massage; 753
going_out_of_room; 910
going_up_or_down_stairs; 892
hanging_out_or_taking_in_laundry; 1178
holding_someone; 691
holding_someone_on_back; 984
housecleaning; 1207
hugging; 460
ironing; 1124
jumping_on_sofa_or_bed; 976
knitting_or_stitching; 551
listening_to_music_with_headphones; 530
lying_on_floor; 1324
manicuring; 1251
opening_or_closing_container; 802
opening_refrigerator_door; 999
operating_remote_control; 1201
passing_something; 685
playing_board_game; 831
playing_computer_game; 1038
polishing_shoe; 1536
pouring_tea_or_coffee; 1210
putting_off_cloth; 987
putting_on_cloth; 964
reading_book; 529
reading_newspaper; 884
sewing; 400
shaking_hands; 972
shaking_head; 480
shaving; 885
sitting_down; 886
sleeping_on_bed; 342
smoking; 402
standing_on_chair_or_table_or_stepladder; 1009
standing_up; 503
studying; 964
taking_photo; 857
telephoning; 965
throwing; 451
throwing_trash; 1051
using_computer; 334
walking_with_stick; 923
washing_dish; 1202
washing_face; 985
washing_hands; 777
watching_TV; 1218
wearing_glass; 1274
wearing_shoes; 1300
wearing_tie; 869
wiping_window; 1172
writing; 649
Videos Totales = 65386
Proceso terminado correctamente.
"""
            
POSIBLES_ACCIONES = {
"assisting_in_getting_up": 0,
"assisting_in_walking": 1,
"baby_crawling": 2,
"baby_crying": 3,
"being_angry": 4,
"being_surprised": 5,
"bottle-feeding_baby": 6,
"bowing": 7,
"brushing_teeth": 8,
"caressing_head": 9,
"changing_baby_diaper": 10,
"clapping_hands": 11,
"crying": 12,
"cutting_food": 13,
"doing_high_five": 14,
"doing_paper-rock-scissors": 15,
"drinking": 16,
"drying_hair_with_blower": 17,
"eating_meal": 18,
"eating_snack": 19,
"entering_room": 20,
"exercising": 21,
"feeding_baby": 22,
"fighting": 23,
"folding_laundry": 24,
"gardening": 25,
"gargling": 26,
"giving_massage": 27,
"going_out_of_room": 28,
"going_up_or_down_stairs": 29,
"hanging_out_or_taking_in_laundry": 30,
"holding_someone": 31,
"holding_someone_on_back": 32,
"housecleaning": 33,
"hugging": 34,
"ironing": 35,
"jumping_on_sofa_or_bed": 36,
"knitting_or_stitching": 37,
"listening_to_music_with_headphones": 38,
"lying_on_floor": 39,
"manicuring": 40,
"opening_or_closing_container": 41,
"opening_refrigerator_door": 42,
"operating_remote_control": 43,
"passing_something": 44,
"playing_board_game": 45,
"playing_computer_game": 46,
"polishing_shoe": 47,
"pouring_tea_or_coffee": 48,
"putting_off_cloth": 49,
"putting_on_cloth": 50,
"reading_book": 51,
"reading_newspaper": 52,
"sewing": 53,
"shaking_hands": 54,
"shaking_head": 55,
"shaving": 56,
"sitting_down": 57,
"sleeping_on_bed": 58,
"smoking": 59,
"standing_on_chair_or_table_or_stepladder": 60,
"standing_up": 61,
"studying": 62,
"taking_photo": 63,
"telephoning": 64,
"throwing": 65,
"throwing_trash": 66,
"using_computer": 67,
"walking_with_stick": 68,
"washing_dish": 69,
"washing_face": 70,
"washing_hands": 71,
"watching_TV": 72,
"wearing_glass": 73,
"wearing_shoes": 74,
"wearing_tie": 75,
"wiping_window": 76,
"writing": 77 }

NUMERO_ACCIONES = 78

def accionValida(accion):
    for k, v in POSIBLES_ACCIONES.items(): 
        if k == accion:
            return True
    return False

# Tenemos 80 objetos, algunos de los cuales no tienen sentido en un escenario en interiores. Esto lo
# filtraria la ontologia, pero podemos seleccionarlos:
objetos_YOLO = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
"fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
"elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
"skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
"tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
"sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
"potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
"microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
"hair drier", "toothbrush"]

# Nos quedamos con 50 objetos de interes:
objetos_validos_YOLO = ["person", "bench", "cat", "dog", "bear", "backpack", "handbag", "tie", 
"suitcase", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
"sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
"potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
"microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
"hair drier", "toothbrush"]



"""

Assisting
    #"assisting_in_getting_up": 0,
    #"assisting_in_walking": 1,
    #"holding_someone": 31,
    #"holding_someone_on_back": 32,
    #"passing_something": 44,

Baby crawling/crying
    #"baby_crawling": 2,
    #"baby_crying": 3,

Drinking
  #"drinking": 16,
  #"pouring_tea_or_coffee": 48,


Eating
    #"cutting_food": 13,
    #"eating_meal": 18,
    #"eating_snack": 19,

Emotional actions
    "being_angry": 4,
    "being_surprised": 5,
    #"crying": 12,


 'Exercising',
    #"bowing": 7,
    #"clapping_hands": 11,
     #"exercising": 21,
    #"fighting": 23,
    #"shaking_hands": 54,
    #"shaking_head": 55,

Feeding/assisting baby', 
    # "bottle-feeding_baby": 6,
    # "changing_baby_diaper": 10,
    # "feeding_baby": 22,

Housework
     "folding_laundry": 24,
     "gardening": 25,
    "hanging_out_or_taking_in_laundry": 30,
    "housecleaning": 33,
    "ironing": 35,
    "polishing_shoe": 47,
    "wiping_window": 76, 

Massage',
     "caressing_head": 9,
    "giving_massage": 27,


Moving
    "entering_room": 20,
    "going_out_of_room": 28,
    "going_up_or_down_stairs": 29,
    #"walking_with_stick": 68,

Opening something
    #"opening_or_closing_container": 41,
    #"opening_refrigerator_door": 42,

Personal care'
      #"brushing_teeth": 8,
     #"drying_hair_with_blower": 17,
    "gargling": 26,
    "manicuring": 40,
    "shaving": 56,    

Playing something',
     #"doing_high_five": 14,
    #"doing_paper-rock-scissors": 15,
    #"playing_board_game": 45,
    #"playing_computer_game": 46,
     #"using_computer": 67,

 
 'Putting cloths',
     #"putting_off_cloth": 49,
     #"putting_on_cloth": 50,

 
Reading/Writing', 
    #"reading_book": 51,
    #"reading_newspaper": 52,
    # "studying": 62,
    #"writing": 77 }    

Resting
     #"jumping_on_sofa_or_bed": 36,
     "lying_on_floor": 39,
    "sitting_down": 57,
    "sleeping_on_bed": 58,
    "watching_TV": 72,

Sewing/Knitting
     "knitting_or_stitching": 37,
     "sewing": 53,


Standing/Smoking
    #"smoking": 59,
    #"standing_on_chair_or_table_or_stepladder": 60,
    #"standing_up": 61,

Throwing something
    "throwing": 65,
    "throwing_trash": 66,
    
    

Using device
     #"listening_to_music_with_headphones": 38,
    #"operating_remote_control": 43,
    "taking_photo": 63,
    "telephoning": 64,

Washing_something
    #"washing_dish": 69,
    #"washing_face": 70,
    #"washing_hands": 71,


Wearing something
    #"wearing_glass": 73,
    #"wearing_shoes": 74,
    #"wearing_tie": 75,


"""

AGRUPAMIENTO_NOMBRE = ['Assisting',
 'Baby crawling/crying',
 'Drinking', 
 'Eating',
 'Emotional actions', 
 'Exercising',
 'Feeding/assisting baby', 
 'Housework',
 'Massage', 
 'Moving', 
 'Opening something',
 'Personal care', 
 'Playing something',
 'Putting cloths',
 'Reading/Writing', 
 'Resting', 
 'Sewing/Knitting',
 'Standing/Smoking', 
 'Throwing something',
 'Using device', 
 'Washing_something',
 'Wearing something']

AGRUPAMIENTO = [[0, 1, 31, 32, 44],
 [2, 3],
 [16, 48], 
 [13, 18, 19],
 [4, 5, 12, 34], 
 [7, 11, 21, 23, 54, 55 ],
 [6, 10, 22],
 [24, 25, 30, 33, 35, 47, 76], 
 [9, 27], 
 [20, 28, 29, 68],
 [41, 42], 
 [8, 17, 26, 40, 56],
 [14, 15, 45,46, 67],
 [49,50],
 [51,52,62, 77], 
 [36, 39, 57, 58, 72],
 [37, 53],
 [59, 60, 61],
 [65, 66],
 [38, 43, 63, 64],
 [69,70,71],
 [73,74,75]]


#POSIBLES_ACCIONES = {
#"assisting_in_getting_up": 0,
#"assisting_in_walking": 1,
#"baby_crawling": 2,
#"baby_crying": 3,
#"being_angry": 4,
#"being_surprised": 5,
# "bottle-feeding_baby": 6,
#"bowing": 7,
#"brushing_teeth": 8,
#"caressing_head": 9,
# "changing_baby_diaper": 10,
#"clapping_hands": 11,
#"crying": 12,
#"cutting_food": 13,
#"doing_high_five": 14,
#"doing_paper-rock-scissors": 15,
#"drinking": 16,
#"drying_hair_with_blower": 17,
#"eating_meal": 18,
#"eating_snack": 19,
#"entering_room": 20,
#"exercising": 21,
# "feeding_baby": 22,
#"fighting": 23,
#"folding_laundry": 24,
#"gardening": 25,
#"gargling": 26,
#"giving_massage": 27,
#"going_out_of_room": 28,
#"going_up_or_down_stairs": 29,
#"hanging_out_or_taking_in_laundry": 30,
#"holding_someone": 31,
#"holding_someone_on_back": 32,
#"housecleaning": 33,
#"hugging": 34,
#"ironing": 35,
#"jumping_on_sofa_or_bed": 36,
#"knitting_or_stitching": 37,
#"listening_to_music_with_headphones": 38,
#"lying_on_floor": 39,
#"manicuring": 40,
#"opening_or_closing_container": 41,
#"opening_refrigerator_door": 42,
#"operating_remote_control": 43,
#"passing_something": 44,
#"playing_board_game": 45,
#"playing_computer_game": 46,
#"polishing_shoe": 47,
#"pouring_tea_or_coffee": 48,
#"putting_off_cloth": 49,
#"putting_on_cloth": 50,
#"reading_book": 51,
#"reading_newspaper": 52,
#"sewing": 53,
#"shaking_hands": 54,
#"shaking_head": 55,
#"shaving": 56,
#"sitting_down": 57,
#"sleeping_on_bed": 58,
#"smoking": 59,
#"standing_on_chair_or_table_or_stepladder": 60,
#"standing_up": 61,
# "studying": 62,
#"taking_photo": 63,
#"telephoning": 64,
#"throwing": 65,
#"throwing_trash": 66,
#"using_computer": 67,
#"walking_with_stick": 68,
#"washing_dish": 69,
#"washing_face": 70,
#"washing_hands": 71,
#"watching_TV": 72,
#"wearing_glass": 73,
#"wearing_shoes": 74,
#"wearing_tie": 75,
#"wiping_window": 76,
#"writing": 77 }
