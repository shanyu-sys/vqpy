
image_resolution = (1513, 854)
video_resolution = (1920, 1080)
side_walk_image = [
    (576, 435), (757, 446),
    (0, 820), (344, 854)
]

right_turn_corner_image = [
    (759, 447), (998, 441), (998, 571), (664, 542)
]

side_walk_video = [(int(x / 1513 * 1920), int(y / 854 * 1080)) for x, y in side_walk_image]
right_turn_corner_video = [(int(x / 1513 * 1920), int(y / 854 * 1080)) for x, y in right_turn_corner_image]


# Query 1: person_side_walk = SpatialQuery(Person, SideWalk, funcion: is_inside)
# Query 2: car_turning = SpatialQuery(Car, RightTurnCorner, funcion: is_inside)
# Query 3: car_waiting = DurationQuery(car_turning, 2s)
# Query 4: car_crossing = SpatialQuery(Car, SideWalk, funcion: is_inside)
# Query 5: car_waiting_person = car_waiting & person_side_walk
# Query 6: TemporalQuery([car_waiting_person, car_crossing])