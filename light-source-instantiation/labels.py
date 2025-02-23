# Adjusted from: https://github.com/mcordts/cityscapesScripts/blob/a7ac7b4062d1a80ed5e22d2ea2179c886801c77d/cityscapesscripts/helpers/labels.py#L3

from collections import namedtuple

#--------------------------------------------------------------------------------
# Definitions
#--------------------------------------------------------------------------------

# a label and all meta information
Label = namedtuple( 'Label' , [

	'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
					# We use them to uniquely name a class

	'id'          , # An integer ID that is associated with this label.
					# The IDs are used to represent the label in ground truth images
					# An ID of -1 means that this label does not have an ID and thus
					# is ignored when creating ground truth images (e.g. license plate).
					# Do not modify these IDs, since exactly these IDs are expected by the
					# evaluation server.

	'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
					# ground truth images with train IDs, using the tools provided in the
					# 'preparation' folder. However, make sure to validate or submit results
					# to our evaluation server using the regular IDs above!
					# For trainIds, multiple labels might have the same ID. Then, these labels
					# are mapped to the same class in the ground truth images. For the inverse
					# mapping, we use the label that is defined first in the list below.
					# For example, mapping all void-type classes to the same ID in training,
					# might make sense for some approaches.
					# Max value is 255!

	'category'    , # The name of the category that this label belongs to

	'categoryId'  , # The ID of this category. Used to create ground truth images
					# on category level.

	'hasInstances', # Whether this label distinguishes between single instances or not

	'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
					# during evaluations or not

	'color'       , # The color of this label

	] )

#--------------------------------------------------------------------------------
# A list of all labels
#--------------------------------------------------------------------------------

# Please adapt the train IDs as appropriate for your approach.
# Note that you might want to ignore labels with ID 255 during training.
# Further note that the current train IDs are only a suggestion. You can use whatever you like.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!

class LabelSet:

	def __init__(self, labels):
		self.labels = labels
		self.name2label      = { label.name    : label for label in labels           }
		self.name2id         = { label.name    : label.id for label in labels        }

		self.id2label        = { label.id      : label for label in labels           }
		self.id2name         = { label.id      : label.name for label in labels      }
		self.id2category     = { label.id      : label.category for label in labels  }

		self.trainId2label   = { label.trainId : label for label in reversed(labels) }
		self.trainId2color   = { label.trainId : label.color for label in labels }

		self.names = [l.name for l in self.labels]

		self.category2labels = {}
		for label in labels:
			category = label.category
			if category in self.category2labels:
				self.category2labels[category].append(label)
			else:
				self.category2labels[category] = [label]

acdc = LabelSet(
				[
					#       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
					Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
					Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
					Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
					Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
					Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
					Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
					Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
					Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
					Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
					Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
					Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
					Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
					Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
					Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
					Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
					Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
					Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
					Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
					Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
					Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
				])

ls = LabelSet(
				[
					#       name                     id    trainId   category       catId     hasInstances   ignoreInEval    color
					Label(  'unlabeled'            ,  0 ,      255 ,  'void'           ,-1       ,False        , True         ,(  0,   0,  0) ),
					Label(  'window_building'      ,  1 ,       -1 ,  ''               ,-1       ,True         , False        ,(  0, 113,188) ),
					Label(  'window_parked'        ,  2        ,-1 ,  ''               ,-1       ,True         , False        ,(255, 127,  0) ),
					Label(  'window_transport'     ,  3        ,-1 ,  ''               ,-1       ,True         , False        ,(190, 190,  0) ),
					Label(  'traffic_light'        ,  4        ,-1 ,  ''               ,-1       ,True         , False        ,(236, 176, 31) ),
					Label(  'street_light_HT'      ,  5        ,-1 ,  ''               ,-1       ,True         , False        ,(161,  19, 46) ),
					Label(  'street_light_LT'      ,  6        ,-1 ,  ''               ,-1       ,True         , False        ,(170,   0,255) ),
					Label(  'parked_front'         ,  7        ,-1 ,  ''               ,-1       ,True         , False        ,(255,   0,  0) ),
					Label(  'parked_rear'          ,  8        ,-1 ,  ''               ,-1       ,True         , False        ,(255, 127,  0) ),
					Label(  'moving_front'         ,  9        ,-1 ,  ''               ,-1       ,True         , False        ,(  0, 255,  0) ),
					Label(  'moving_rear'          ,  10       ,-1 ,  ''               ,-1       ,True         , False        ,(  0,   0,255) ),
					Label(  'advertisement'        ,  11       ,-1 ,  ''               ,-1       ,True         , False        ,(255,   0,  0) ),
					Label(  'clock'                ,  12       ,-1 ,  ''               ,-1       ,True         , False        ,( 84,  84,  0) ),
					Label(  'inferred'             ,  13       ,-1 ,  ''               ,-1       ,True         , False        ,( 84, 170,  0) ),
					Label(  'windows_group'        ,  14       ,-1 ,  ''               ,-1       ,True         , False        ,( 84, 255,  0) ),
					Label(  'car'                  ,  15       ,-1 ,  ''               ,-1       ,True         , False        ,(170,  84,  0) ),
					Label(  'truck'                ,  16       ,-1 ,  ''               ,-1       ,True         , False        ,(170, 170,  0) ),
					Label(  'bus'                  ,  17       ,-1 ,  ''               ,-1       ,True         , False        ,(170, 255,  0) ),
					Label(  'bicycle'              ,  18       ,-1 ,  ''               ,-1       ,True         , False        ,(255,  84,  0) ),
					Label(  'motorcycle'           ,  19       ,-1 ,  ''               ,-1       ,True         , False        ,(255, 170,  0) ),
					Label(  'train'                ,  20       ,-1 ,  ''               ,-1       ,True         , False        ,(255, 255,  0) ),
					Label(  'traffic_light_G'      ,  21       ,-1 ,  ''               ,-1       ,True         , False        ,(255, 255,  1) ),
					Label(  'traffic_light_O'      ,  22       ,-1 ,  ''               ,-1       ,True         , False        ,(255, 255,  2) ),
					Label(  'traffic_light_R'      ,  23       ,-1 ,  ''               ,-1       ,True         , False        ,(255, 255,  3) ),
				]) # entries with -1 are values to be set.

vehicles = [
	  'car'
	, 'truck'
	, 'train'
	, 'bus'
	, 'bicycle'
	, 'motorcycle'
]

head_lights = [
	  'moving_front'
	, 'moving_rear'
	, 'parked_front'
	, 'parked_rear'
]

metal = [ # from acdc annotations.
		 *vehicles
		, 'pole'
		, 'traffic sign'

]

glass = [ # from light-source annotations.
		 'window_parked'
		,'window_transport'
		,'window_building'

		,'traffic_light_G'
		,'traffic_light_O'
		,'traffic_light_R'

		,'street_light_HT'
		,'street_light_LT'

		, *head_lights
]

asphalt = [ # from acdc annotations.
		  'sidewalk'
		, 'road'
]


groups = ['windows_group'] + vehicles
sources = list(set(ls.names) - set(groups) - set(['unlabeled']))

# where a source can be mapped to.
source_groups = {
			  "unlabeled": []
			, "window_building": ['windows_group']
			, "window_parked": [*vehicles]
			, "window_transport": ['train', 'bus']
			, "traffic_light_G": []
			, "traffic_light_O": []
			, "traffic_light_R": []
			, "street_light_HT": []
			, "street_light_LT": []
			, "parked_front": [*vehicles]
			, "parked_rear": [*vehicles]
			, "moving_front": [*vehicles]
			, "moving_rear": [*vehicles]
			, "advertisement": []
			, "clock": []
			, "inferred": ['train', 'bus']
}

def get_uniform_params(*keys):
	"""
		Given a list of keys the multi-level dictionary of uniform parameters is indexed.
	"""

	key_path = list(keys)
	
	def get_v(param_dict, key_path):
		if len(key_path) == 1:
			if param_dict is None:
				return None

			return param_dict.get(key_path[0])

		return get_v(param_dict.get(key_path[0]), key_path[1:])

	params = {
		'root': {
			'inferred':(1, 1)
			,'traffic_light_G':(1, 1)
			,'traffic_light_R':(1, 1)
			,'traffic_light_O':(1, 1)
			,'street_light_HT':(1, 1)
			,'street_light_LT':(1, 1)
			,'advertisement':(.6,.8)
			,'clock':(.8, 1)
			,'window_building':(.3, .6)

			,'windows_group': {
				'window_building':(.3, .6)
			}
			,'car': {
				'window_parked':(.1,.4)
				,'moving_front': (.95,1)
				,'moving_rear': (.95,1)
				,'parked_front': (.1,.3)
				,'parked_rear': (.1,.3)
			}
			,'bus': {
				'window_parked':(.1, .4)
				,'moving_front': (.95,1)
				,'moving_rear': (.95,1)
				,'parked_front': (.1,.3)
				,'parked_rear': (.1,.3)
				,'window_transport':(.9, 1)
				,'inferred':(1, 1)
			}
			,'train': {
				'moving_front': (.95,1)
				,'moving_rear': (.95,1)
				,'parked_front': (.1,.3)
				,'parked_rear': (.1,.3)
				,'window_transport':(.9, 1)
				,'inferred':(1, 1)
			}
			,'truck': {
				'window_parked':(.1, .4)
				,'moving_front': (.95,1)
				,'moving_rear': (.95,1)
				,'parked_front': (.1,.3)
				,'parked_rear': (.1,.3)
			}
			,'motorcycle': {
				'moving_front': (.95,1)
				,'moving_rear': (.95,1)
				,'parked_front': (.1,.3)
				,'parked_rear': (.1,.3)
			}
			,'bicycle': {
				'moving_front': (.95,1)
				,'moving_rear': (.95,1)
				,'parked_front': (.1,.2)
				,'parked_rear': (.1,.2)
			}
		}
	}

	return get_v(params, key_path)


strength = {

	  'window_building':  (1, 20)

	, 'window_parked':    (1, 2)
	, 'window_transport': (8, 10)

	, 'traffic_light_G':  (90, 250)
	, 'traffic_light_O':  (90, 250)
	, 'traffic_light_R':  (90, 250)

	, 'parked_rear':      (60, 90)
	, 'moving_rear':      (60, 90)

	, 'street_light_HT':  (850, 1000)
	, 'street_light_LT':  (850, 1000)

	, 'parked_front':     (750, 900)
	, 'moving_front':     (750, 900)

	# not supported!
	, 'advertisement':    (0.5, 1)
	, 'clock':            (0.5, 1)
	, 'inferred':         (0.5, 1)
}

street_lights = [
		  'street_light_HT'
		, 'street_light_LT'
]



# Filename_ids to be exluded in the light source mask generation. (=1)
fn_id_exclusion_list = []
# 'GP020397_frame_000345' # window_building labeled as traffic_light.


# light sources annotations that have been created manually.
manual_annotations = [
  'GOPR0351_frame_000033'
, 'GOPR0351_frame_000043'
, 'GOPR0351_frame_000049'
, 'GOPR0351_frame_000055'
, 'GOPR0351_frame_000059'
, 'GOPR0351_frame_000063'
, 'GOPR0351_frame_000067'
, 'GOPR0351_frame_000071'
, 'GOPR0351_frame_000081'
, 'GOPR0351_frame_000085'
, 'GOPR0351_frame_000089'
, 'GOPR0351_frame_000093'
, 'GOPR0351_frame_000099'
, 'GOPR0351_frame_000105'
, 'GOPR0351_frame_000113'
, 'GOPR0351_frame_000121'
, 'GOPR0351_frame_000128'
, 'GOPR0351_frame_000141'
, 'GOPR0351_frame_000159'
, 'GOPR0351_frame_000165'
, 'GOPR0351_frame_000169'
, 'GOPR0351_frame_000177'
, 'GOPR0351_frame_000181'
, 'GOPR0351_frame_000185'
, 'GOPR0351_frame_000191'
, 'GOPR0351_frame_000198'
, 'GOPR0351_frame_000202'
, 'GOPR0351_frame_000214'
, 'GOPR0351_frame_000226'
, 'GOPR0351_frame_000233'
, 'GOPR0351_frame_000253'
, 'GOPR0351_frame_000260'
, 'GOPR0351_frame_000267'
, 'GOPR0351_frame_000276'
, 'GOPR0351_frame_000285'
, 'GOPR0351_frame_000304'
, 'GOPR0351_frame_000313'
, 'GOPR0351_frame_000319'
, 'GOPR0351_frame_000325'
, 'GOPR0351_frame_000329'
, 'GOPR0351_frame_000335'
, 'GOPR0351_frame_000341'
, 'GOPR0351_frame_000354'
, 'GOPR0351_frame_000376'
, 'GOPR0351_frame_000382'
, 'GOPR0351_frame_000388'
, 'GOPR0351_frame_000396'
, 'GOPR0351_frame_000405'
, 'GOPR0351_frame_000487'
, 'GOPR0351_frame_000494'
, 'GOPR0351_frame_000500'
, 'GOPR0351_frame_000504'
, 'GOPR0351_frame_000509'
, 'GOPR0351_frame_000515'
, 'GOPR0351_frame_000534'
, 'GOPR0351_frame_000543'
, 'GOPR0351_frame_000547'
, 'GOPR0351_frame_000553'
, 'GOPR0351_frame_000557'
, 'GOPR0351_frame_000581'
, 'GOPR0351_frame_000599'
, 'GOPR0351_frame_000607'
, 'GOPR0351_frame_000619'
, 'GOPR0351_frame_000627'
, 'GOPR0351_frame_000639'
, 'GOPR0351_frame_000651'
, 'GOPR0351_frame_000706'
, 'GOPR0351_frame_000712'
, 'GOPR0351_frame_000724'
, 'GOPR0351_frame_000742'
, 'GOPR0351_frame_000748'
, 'GOPR0351_frame_000761'
, 'GOPR0351_frame_000774'
, 'GOPR0351_frame_000814'
, 'GOPR0351_frame_000822'
, 'GOPR0356_frame_000321'
, 'GOPR0356_frame_000327'
, 'GOPR0356_frame_000333'
, 'GOPR0356_frame_000339'
, 'GOPR0356_frame_000345'
, 'GOPR0356_frame_000351'
, 'GOPR0356_frame_000357'
, 'GOPR0356_frame_000363'
, 'GOPR0356_frame_000369'
, 'GOPR0356_frame_000375'
, 'GOPR0356_frame_000382'
, 'GOPR0356_frame_000391'
, 'GOPR0356_frame_000398'
, 'GOPR0356_frame_000404'
, 'GOPR0356_frame_000414'
, 'GOPR0356_frame_000421'
, 'GOPR0356_frame_000427'
, 'GOPR0356_frame_000433'
, 'GOPR0356_frame_000448'
, 'GOPR0356_frame_000455'
, 'GOPR0356_frame_000461'
, 'GOPR0356_frame_000467'
, 'GOPR0356_frame_000473'
, 'GOPR0356_frame_000479'
, 'GOPR0356_frame_000485'
, 'GOPR0376_frame_000038'
, 'GOPR0376_frame_000046'
, 'GOPR0376_frame_000054'
, 'GOPR0376_frame_000067'
, 'GOPR0376_frame_000097'
, 'GOPR0376_frame_000105'
, 'GOPR0376_frame_000113'
, 'GOPR0376_frame_000122'
, 'GOPR0376_frame_000146'
, 'GOPR0376_frame_000157'
, 'GOPR0376_frame_000165'
, 'GOPR0376_frame_000173'
, 'GOPR0376_frame_000181'
, 'GOPR0376_frame_000189'
, 'GOPR0376_frame_000198'
, 'GOPR0376_frame_000229'
, 'GOPR0376_frame_000268'
, 'GOPR0376_frame_000276'
, 'GOPR0376_frame_000284'
, 'GOPR0376_frame_000292'
, 'GOPR0376_frame_000302'
, 'GOPR0376_frame_000310'
, 'GOPR0376_frame_000324'
, 'GOPR0376_frame_000332'
, 'GOPR0376_frame_000340'
, 'GOPR0376_frame_000349'
, 'GOPR0376_frame_000369'
, 'GOPR0376_frame_000382'
, 'GOPR0376_frame_000418'
, 'GOPR0376_frame_000435'
, 'GOPR0376_frame_000466'
, 'GOPR0376_frame_000478'
, 'GOPR0376_frame_000486'
, 'GOPR0376_frame_000496'
, 'GOPR0376_frame_000511'
, 'GOPR0376_frame_000533'
, 'GOPR0376_frame_000548'
, 'GOPR0376_frame_000565'
, 'GOPR0376_frame_000576'
, 'GOPR0376_frame_000590'
, 'GOPR0376_frame_000608'
, 'GOPR0376_frame_000621'
, 'GOPR0376_frame_000633'
, 'GOPR0376_frame_000650'
, 'GOPR0376_frame_000663'
, 'GOPR0376_frame_000699'
, 'GOPR0376_frame_000721'
, 'GOPR0376_frame_000733'
, 'GOPR0376_frame_000749'
, 'GOPR0376_frame_000764'
, 'GOPR0376_frame_000788'
, 'GOPR0376_frame_000826'
, 'GOPR0376_frame_000843'
, 'GOPR0376_frame_000851'
, 'GOPR0376_frame_000862'
, 'GOPR0376_frame_000870'
, 'GOPR0376_frame_000878'
, 'GOPR0376_frame_000887'
, 'GOPR0376_frame_000916'
, 'GOPR0376_frame_000927'
, 'GOPR0376_frame_000963'
, 'GOPR0376_frame_000971'
, 'GOPR0376_frame_000979'
, 'GOPR0376_frame_000991'
, 'GOPR0376_frame_001013'
, 'GOPR0376_frame_001021'
, 'GOPR0376_frame_001054'
, 'GOPR0475_frame_000041'
, 'GOPR0475_frame_000096'
, 'GOPR0475_frame_000122'
, 'GOPR0475_frame_000135'
, 'GOPR0475_frame_000154'
, 'GOPR0475_frame_000168'
, 'GOPR0475_frame_000180'
, 'GOPR0475_frame_000191'
, 'GOPR0475_frame_000203'
, 'GOPR0475_frame_000215'
, 'GOPR0476_frame_000001'
, 'GOPR0476_frame_000028'
, 'GOPR0476_frame_000056'
, 'GOPR0476_frame_000076'
, 'GOPR0476_frame_000084'
, 'GOPR0476_frame_000094'
, 'GOPR0476_frame_000102'
, 'GOPR0476_frame_000127'
, 'GOPR0476_frame_000137'
, 'GOPR0476_frame_000152'
, 'GOPR0476_frame_000180'
, 'GOPR0476_frame_000241'
, 'GOPR0476_frame_000251'
, 'GOPR0476_frame_000259'
, 'GOPR0476_frame_000284'
, 'GOPR0476_frame_000311'
, 'GOPR0476_frame_000351'
, 'GOPR0476_frame_000362'
, 'GOPR0476_frame_000382'
, 'GOPR0476_frame_000394'
, 'GOPR0476_frame_000416'
, 'GOPR0476_frame_000429'
, 'GOPR0476_frame_000456'
, 'GOPR0476_frame_000469'
, 'GOPR0476_frame_000481'
, 'GOPR0476_frame_000508'
, 'GOPR0476_frame_000523'
, 'GOPR0476_frame_000594'
, 'GOPR0477_frame_000010'
, 'GOPR0477_frame_000025'
, 'GOPR0477_frame_000053'
, 'GOPR0477_frame_000062'
, 'GOPR0477_frame_000075'
, 'GOPR0477_frame_000087'
, 'GOPR0477_frame_000099'
, 'GOPR0477_frame_000107'
, 'GOPR0477_frame_000214'
, 'GOPR0477_frame_000223'
, 'GOPR0477_frame_000231'
, 'GOPR0477_frame_000255'
, 'GOPR0477_frame_000323'
, 'GOPR0477_frame_000329'
, 'GOPR0477_frame_000338'
, 'GOPR0477_frame_000344'
, 'GOPR0477_frame_000368'
, 'GOPR0477_frame_000378'
, 'GOPR0477_frame_000387'
, 'GOPR0477_frame_000459'
, 'GOPR0477_frame_000468'
, 'GOPR0477_frame_000474'
, 'GOPR0477_frame_000480'
, 'GOPR0477_frame_000489'
, 'GOPR0477_frame_000498'
, 'GOPR0477_frame_000505'
, 'GOPR0477_frame_000516'
, 'GOPR0477_frame_000529'
, 'GOPR0477_frame_000537'
, 'GOPR0477_frame_000545'
, 'GOPR0477_frame_000553'
, 'GOPR0477_frame_000561'
, 'GOPR0477_frame_000571'
, 'GOPR0477_frame_000599'
, 'GOPR0477_frame_000607'
, 'GOPR0477_frame_000614'
, 'GOPR0477_frame_000621'
, 'GOPR0477_frame_000633'
, 'GOPR0477_frame_000665'
, 'GOPR0477_frame_000673'
, 'GOPR0477_frame_000682'
, 'GOPR0477_frame_000700'
, 'GOPR0477_frame_000711'
, 'GOPR0477_frame_000722'
, 'GOPR0477_frame_000732'
, 'GOPR0477_frame_000742'
, 'GOPR0477_frame_000751'
, 'GOPR0477_frame_000761'
, 'GOPR0477_frame_000770'
, 'GOPR0477_frame_000778'
, 'GOPR0477_frame_000786'
, 'GOPR0478_frame_000017'
, 'GOPR0478_frame_000025'
, 'GOPR0478_frame_000031'
, 'GOPR0478_frame_000037'
, 'GOPR0478_frame_000045'
, 'GOPR0478_frame_000055'
, 'GOPR0478_frame_000095'
, 'GOPR0478_frame_000132'
, 'GOPR0478_frame_000140'
, 'GOPR0478_frame_000148'
, 'GOPR0478_frame_000159'
, 'GOPR0478_frame_000182'
, 'GOPR0478_frame_000191'
, 'GOPR0478_frame_000199'
, 'GOPR0478_frame_000207'
, 'GOPR0478_frame_000215'
, 'GOPR0478_frame_000223'
, 'GOPR0478_frame_000231'
, 'GOPR0478_frame_000239'
, 'GOPR0478_frame_000247'
, 'GOPR0478_frame_000255'
, 'GOPR0479_frame_000033'
, 'GOPR0479_frame_000047'
, 'GOPR0479_frame_000077'
, 'GOPR0479_frame_000125'
, 'GOPR0479_frame_000131'
, 'GOPR0479_frame_000143'
, 'GOPR0479_frame_000151'
, 'GOPR0479_frame_000159'
, 'GOPR0479_frame_000165'
, 'GOPR0479_frame_000173'
, 'GP010376_frame_000001'
, 'GP010376_frame_000009'
, 'GP010376_frame_000020'
, 'GP010376_frame_000028'
, 'GP010376_frame_000036'
, 'GP010376_frame_000044'
, 'GP010376_frame_000084'
, 'GP010376_frame_000129'
, 'GP010376_frame_000137'
, 'GP010376_frame_000145'
, 'GP010376_frame_000153'
, 'GP010397_frame_000037'
, 'GP010397_frame_000063'
, 'GP010397_frame_000080'
, 'GP010397_frame_000102'
, 'GP010397_frame_000166'
, 'GP010397_frame_000176'
, 'GP010397_frame_000193'
, 'GP010397_frame_000211'
, 'GP010397_frame_000222'
, 'GP010397_frame_000337'
, 'GP010397_frame_000411'
, 'GP010397_frame_000464'
, 'GP010397_frame_000558'
, 'GP010397_frame_000577'
, 'GP010397_frame_000630'
, 'GP010397_frame_000663'
, 'GP010397_frame_000697'
, 'GP010397_frame_000728'
, 'GP010397_frame_000775'
, 'GP010397_frame_000800'
, 'GP010397_frame_000835'
, 'GP010397_frame_000864'
, 'GP010397_frame_000902'
, 'GP010397_frame_000910'
, 'GP010397_frame_000938'
, 'GP010397_frame_000950'
, 'GP010397_frame_000970'
, 'GP010397_frame_001006'
, 'GP010397_frame_001019'
, 'GP010397_frame_001042'
, 'GP020397_frame_000001'
, 'GP020397_frame_000042'
, 'GP020397_frame_000079'
, 'GP020397_frame_000111'
, 'GP020397_frame_000121'
, 'GP020397_frame_000167'
, 'GP020397_frame_000191'
, 'GP020397_frame_000236'
, 'GP020397_frame_000256'
, 'GP020397_frame_000266'
, 'GP020397_frame_000286'
, 'GP020397_frame_000322'
, 'GP020397_frame_000330'
, 'GP020397_frame_000345'
, 'GP020397_frame_000385'
, 'GP020397_frame_000410'
, 'GP020397_frame_000425'
, 'GP020397_frame_000448'
, 'GP020397_frame_000479'
, 'GP020397_frame_000531'
, 'GP020397_frame_000611'
, 'GP020397_frame_000643'
]
