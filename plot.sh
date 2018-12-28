#!/bin/bash

echo "======== PLOTTING BASELINES ========"
python3 plot.py --mode baselines --in_dir data/baselines/HalfCheetah-v2 --env HalfCheetah-v2 --avg_window 200
python3 plot.py --mode baselines --in_dir data/baselines/Humanoid-v2 --env Humanoid-v2 --avg_window 2000
python3 plot.py --mode baselines --in_dir data/baselines/Swimmer-v2 --env Swimmer-v2 --avg_window 100
python3 plot.py --mode baselines --in_dir data/baselines/Walker2d-v2 --env Walker2d-v2 --avg_window 1500

mv plots/HalfCheetah-v2.jpg plots/HalfCheetah-v2-baselines.jpg
mv plots/Humanoid-v2.jpg plots/Humanoid-v2-baselines.jpg
mv plots/Swimmer-v2.jpg plots/Swimmer-v2-baselines.jpg
mv plots/Walker2d-v2.jpg plots/Walker2d-v2-baselines.jpg

echo "======== PLOTTING RNNS ========"
python3 plot.py --mode rnns --in_dir data/rnns/HalfCheetah-v2 --env HalfCheetah-v2 --avg_window 200
python3 plot.py --mode rnns --in_dir data/rnns/Humanoid-v2 --env Humanoid-v2 --avg_window 2000
python3 plot.py --mode rnns --in_dir data/rnns/Swimmer-v2 --env Swimmer-v2 --avg_window 100
python3 plot.py --mode rnns --in_dir data/rnns/Walker2d-v2 --env Walker2d-v2 --avg_window 1500

mv plots/HalfCheetah-v2.jpg plots/HalfCheetah-v2-rnns.jpg
mv plots/Humanoid-v2.jpg plots/Humanoid-v2-rnns.jpg
mv plots/Swimmer-v2.jpg plots/Swimmer-v2-rnns.jpg
mv plots/Walker2d-v2.jpg plots/Walker2d-v2-rnns.jpg

echo "======== PLOTTING RCN BIASES ========"
python3 plot.py --mode rcn-biases --in_dir data/rcn-biases/HalfCheetah-v2 --env HalfCheetah-v2 --avg_window 200
python3 plot.py --mode rcn-biases --in_dir data/rcn-biases/Humanoid-v2 --env Humanoid-v2 --avg_window 2000
python3 plot.py --mode rcn-biases --in_dir data/rcn-biases/Swimmer-v2 --env Swimmer-v2 --avg_window 100
python3 plot.py --mode rcn-biases --in_dir data/rcn-biases/Walker2d-v2 --env Walker2d-v2 --avg_window 1500

mv plots/HalfCheetah-v2.jpg plots/HalfCheetah-v2-rcn-biases.jpg
mv plots/Humanoid-v2.jpg plots/Humanoid-v2-rcn-biases.jpg
mv plots/Swimmer-v2.jpg plots/Swimmer-v2-rcn-biases.jpg
mv plots/Walker2d-v2.jpg plots/Walker2d-v2-rcn-biases.jpg
