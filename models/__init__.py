import logging

def create_model(opt):
	model = None
	if opt.model == 'dr_uspv': 
		# assert(opt.dataset_mode == 'unaligned')
		from .dr_uspv_model import DR_USPV_MODEL
		model = DR_USPV_MODEL()
	elif opt.model == 'test':
		from .test_model import TestModel
		model = TestModel()
	else:
		raise NotImplementedError('model [%s] not implemented.' % opt.model)
	model.initialize(opt)
	logging.info("model [%s] was created" % (model.name()))
	return model
