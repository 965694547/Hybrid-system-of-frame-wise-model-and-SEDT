import numpy as np
from src import trainer

if __name__=='__main__':
    SED_model = trainer.trainer('DCASE2019-task4', 'sed_with_cATP-DF', True, True)
    SED_lst = np.load('example_ids.npy')
    SED_lst = ['merge_separated_' + SED_l for SED_l in SED_lst]
    tags, _ = SED_model.pro_sedt(SED_lst)
    np.save('tags.npy', tags)
