from ppf import ppf
from voting import voting
import time

oModelPPF = ppf("bolt_model.pcd", 4, 0.05, 25)

oModelPPF.vis()
oModelPPF.make_hash_table()
oModelPPF.save_hashtable("model-bolt.txt")




oScenePPF = ppf("bolt_model.pcd", 5, 0.05, 25)
# oScenePPF.vis()

oVoting = voting(oModelPPF, oScenePPF, 20)
