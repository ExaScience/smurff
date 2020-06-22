commit a7505284916dc993f63c6ae499bc2373b8d6f7ec
Author: Tom Vander Aa <vanderaa@imec.be>
Date:   Thu Jun 18 15:27:39 2020 +0200

    FIX: pred_avg/pred_var for Tensors

diff --git a/python/smurff/smurff/predict.py b/python/smurff/smurff/predict.py
index 40d57a08..7eded23c 100755
--- a/python/smurff/smurff/predict.py
+++ b/python/smurff/smurff/predict.py
@@ -43,8 +43,15 @@ class Sample:
 
         # predictions, rmse
         sample.pred_stats = dict(read_config_file(cp["predictions"]["pred_state"], dir_name)["global"].items())
-        sample.pred_avg = mio.read_matrix(os.path.join(dir_name, cp["predictions"]["pred_avg"]))
-        sample.pred_var = mio.read_matrix(os.path.join(dir_name, cp["predictions"]["pred_var"]))
+        if "pred_avg" in cp["predictions"]:
+            sample.pred_avg = mio.read_matrix(os.path.join(dir_name, cp["predictions"]["pred_avg"]))
+        else:
+            sample.pred_avg = None
+
+        if "pred_var" in cp["predictions"]:
+            sample.pred_var = mio.read_matrix(os.path.join(dir_name, cp["predictions"]["pred_var"]))
+        else:
+            sample.pred_var = None
 
         # latent matrices
         for i in range(sample.nmodes):
@@ -80,7 +87,6 @@ class Sample:
         return sample
 
     def __init__(self, nmodes, it):
-        assert nmodes == 2
         self.nmodes = nmodes
         self.iter = it
         self.latents = []
@@ -172,7 +178,6 @@ class PredictSession:
         self.options = read_config_file(cp["options"]["options"], self.root_dir)
 
         self.nmodes = self.options.getint("global", "num_priors")
-        assert self.nmodes == 2
 
         # load only one sample
         for step_name, step_file in cp["steps"].items():
