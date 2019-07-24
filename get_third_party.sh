#!/bin/bash
set -x
set -e

mkdir -p third_party
cd third_party

# ================= MOSES =====================
git clone 'https://github.com/moses-smt/mosesdecoder.git' moses
pushd .
cd moses
git checkout RELEASE-4.0
popd


# ================= FASTTEXT ==================
git clone 'https://github.com/facebookresearch/fastText.git' fastText
pushd .
cd fastText
git checkout v0.2.0
make
popd

# ================= MUSE ======================
# TODO use facebook's repo when this is merged (pull requests #81 or #97)
git clone 'https://github.com/gitlost-murali/MUSE.git' MUSE
pushd .
cd MUSE
git checkout Hi-Ur-unsupervisedMT
echo -e "diff --git a/src/trainer.py b/src/trainer.py\nindex b9d4444..dfe2446 100644\n--- a/src/trainer.py\n+++ b/src/trainer.py\n@@ -90,7 +90,7 @@ class Trainer(object):\n         x, y = self.get_dis_xy(volatile=True)\n         preds = self.discriminator(Variable(x.data))\n         loss = F.binary_cross_entropy(preds, y)\n-        stats['DIS_COSTS'].append(loss.data[0])\n+        stats['DIS_COSTS'].append(loss.data.item())\n \n         # check NaN\n         if (loss != loss).data.any():" | git apply
popd
