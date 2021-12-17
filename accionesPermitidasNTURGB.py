# Acciones de NTU-RGB
# Autor: Jaime Duque Domingo
""" 
Acciones de NTU-RGB:
A1. drink water.
A2. eat meal/snack.
A3. brushing teeth.
A4. brushing hair.
A5. drop.
A6. pickup.
A7. throw.
A8. sitting down.
A9. standing up (from sitting position).
A10. clapping.
A11. reading.
A12. writing.
A13. tear up paper.
A14. wear jacket.
A15. take off jacket.
A16. wear a shoe.
A17. take off a shoe.
A18. wear on glasses.
A19. take off glasses.
A20. put on a hat/cap.
A21. take off a hat/cap.
A22. cheer up.
A23. hand waving.
A24. kicking something.
A25. reach into pocket.
A26. hopping (one foot jumping).
A27. jump up.
A28. make a phone call/answer phone.
A29. playing with phone/tablet.
A30. typing on a keyboard.
A31. pointing to something with finger.
A32. taking a selfie.
A33. check time (from watch).
A34. rub two hands together.
A35. nod head/bow.
A36. shake head.
A37. wipe face.
A38. salute.
A39. put the palms together.
A40. cross hands in front (say stop).
A41. sneeze/cough.
A42. staggering.
A43. falling.
A44. touch head (headache).
A45. touch chest (stomachache/heart pain).
A46. touch back (backache).
A47. touch neck (neckache).
A48. nausea or vomiting condition.
A49. use a fan (with hand or paper)/feeling warm.
A50. punching/slapping other person.
A51. kicking other person.
A52. pushing other person.
A53. pat on back of other person.
A54. point finger at the other person.
A55. hugging other person.
A56. giving something to other person.
A57. touch other person's pocket.
A58. handshaking.
A59. walking towards each other.
A60. walking apart from each other.   


NTURGBD-120 tiene 120 acciones 
"""
            
POSIBLES_ACCIONES = {
"A001": 0,
"A002": 1,
"A003": 2,
"A004": 3,
"A005": 4,
"A006": 5,
"A007": 6,
"A008": 7,
"A009": 8,
"A010": 9,
"A011": 10,
"A012": 11,
"A013": 12,
"A014": 13,
"A015": 14,
"A016": 15,
"A017": 16,
"A018": 17,
"A019": 18,
"A020": 19,
"A021": 20,
"A022": 21,
"A023": 22,
"A024": 23,
"A025": 24,
"A026": 25,
"A027": 26,
"A028": 27,
"A029": 28,
"A030": 29,
"A031": 30,
"A032": 31,
"A033": 32,
"A034": 33,
"A035": 34,
"A036": 35,
"A037": 36,
"A038": 37,
"A039": 38,
"A040": 39,
"A041": 40,
"A042": 41,
"A043": 42,
"A044": 43,
"A045": 44,
"A046": 45,
"A047": 46,
"A048": 47,
"A049": 48,
"A050": 49,
"A051": 50,
"A052": 51,
"A053": 52,
"A054": 53,
"A055": 54,
"A056": 55,
"A057": 56,
"A058": 57,
"A059": 58,
"A060": 59,
"A061": 60,
"A062": 61,
"A063": 62,
"A064": 63,
"A065": 64,
"A066": 65,
"A067": 66,
"A068": 67,
"A069": 68,
"A070": 69,
"A071": 70,
"A072": 71,
"A073": 72,
"A074": 73,
"A075": 74,
"A076": 75,
"A077": 76,
"A078": 77,
"A079": 78,
"A080": 79,
"A081": 80,
"A082": 81,
"A083": 82,
"A084": 83,
"A085": 84,
"A086": 85,
"A087": 86,
"A088": 87,
"A089": 88,
"A090": 89,
"A091": 90,
"A092": 91,
"A093": 92,
"A094": 93,
"A095": 94,
"A096": 95,
"A097": 96,
"A098": 97,
"A099": 98,
"A100": 99,
"A101": 100,
"A102": 101,
"A103": 102,
"A104": 103,
"A105": 104,
"A106": 105,
"A107": 106,
"A108": 107,
"A109": 108,
"A110": 109,
"A111": 110,
"A112": 111,
"A113": 112,
"A114": 113,
"A115": 114,
"A116": 115,
"A117": 116,
"A118": 117,
"A119": 118,
"A120": 119}

NUMERO_ACCIONES = 120

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
