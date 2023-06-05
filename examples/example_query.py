import vqpy

class Car(vqpy.vobj.Vehicle):
	# license_plate is a property in Vehicle (VQPy library)

	@stateful(input="bbox", history_len=2)
	def direction(self, hist_bboxes):
	  if hist_bboxes[0][0] - hist_bboxes[1][0] > 0:
		  return "up"
	  else: 
		  return "other directions"

	@stateful(input="coordinate", history_len=2)
	def speed(self, coords):
		pass

	@stateless(input="image")
	def license_plate(self, car_image):
		from openalpr import Alpr
		return Alpr().recognize(car_image)

	@stateless(input="image")
	def color(self, car_image):
	  from colordetect import ColorDetect # python lib
	  return ColorDetect(car_image).result()

class FindRedCar(vqpy.Query):
  def __init__():
	 self.car = Car()

  def frame_constraint():
	 return self.car.color == "red"

  def frame_output():
   return self.car.bbox
  
class FindAmberAlertCar(FindRedCar):
	def frame_constraint():
		return super().frame_constraint() & \
			(self.car.make == "Honda") & \
			(self.car.license_plate.startswith("ABC"))


class SpatialRelation:
	def __init__(self, vobj1, vobj2):
		self.vobj1 = vobj1
		self.vobj2 = vobj2

	@stateless(input1="bbox", input2="bbox")
	def direction(self, bboxes):
		if bboxes[0][0] - bboxes[1][0] > 0:
			return "left"
		else:
			return "right"
		
	@stateless(input1="bbox", input2="bbox")     
	def inside(self, bboxes):
		return bboxes[0][0] < bboxes[1][0] and bboxes[0][1] < bboxes[1][1] and \
			bboxes[0][2] > bboxes[1][2] and bboxes[0][3] > bboxes[1][3]
		
	@stateless(input1="center", input2="center")
	def distance(self, centers):
		dcenter = (centers[0] - centers[1])
		return math.sqrt(sum(dcenter ** 2))
	
	@stateful(input="distance", history_len=2)
	def getting_close(self, dists):
		return dists[0] - dists[1] < 0


class PersonBallRelation(SpatialRelation):
   def __init__(self, person, ball):
	  self.person = person
	  self.ball = ball

	@stateful(input="inside", history_len=5)
	def holding_ball(self, insides):
	   return all(insides)

class FindCloseCar(vqpy.Query):
	def __init__(self):
		self.car1 = Car("car1")
		self.car2 = Car("car2")
		self.spatial_relation = SpatialRelation(self.car1, self.car2)

	def frame_constraint():
		return self.spatial_relation.getting_close

	def frame_output():
		return (self.car1.speed, self.car2.speed)
	
class FindAboutToCrash(FindCloseCar):

	def frame_constraint():
		return super().frame_constraint() & (self.spatial_relation.distance < 10) & (self.car1.speed > 30) | (self.car2.speed > 30)

	def frame_output():
		return (self.car1.speed, self.car2.speed)
	

class TemporalRelation:
	def __init__(self, vobj1, vobj2):
		self.vobj1 = vobj1
		self.vobj2 = vobj2

	def overlap(self):
		return self.vobj1.time_range[0] \
	< self.vobj2.time_range[1] and \
			self.vobj1.time_range[1] > self.vobj2.time_range[0]
   
	def happen_before(self):
		return self.vobj1.time_range[1] < self.vobj2.time_range[0]
	

from vqpy.lib.vobj import Person, Car
from vqpy.lib.query import CollisionQuery, SpeedQuery from vqpy.lib.query import TemporalQuery
query HitAndRun(TemporalQuery): def __init__():
self.car = Car()
self.person = Person()
car_hit_person = CollisionQuery(car, person, dist_threshold) car_run_away = SpeedQuery(car, velocity_threshold) super.__init__(
subqueries=[car_hit_person, car_run_away], time_window="10s", id_match=(car_hit_person.car, car_run_away.car)
)
def video_output():
return self.car.license_plate