def apart_type(indices, room_types, live_rooms):
    apartment_type = ''
    c=0

    for i in indices:
        if room_types[i] in live_rooms:
            c += 1

    if c == 0:
        print('error, c = 0')
    
    elif c == 1:
        apartment_type = 'однушка'
    
    elif c == 2:
        apartment_type = 'двушка'

    elif c == 3:
        apartment_type = 'трёшка'

    elif c == 4:
        apartment_type = '4-х комнатная'

    elif c == 5:
        apartment_type = '5-и комнатная'
    
    else: 
        apartment_type = 'дворец'

    return apartment_type