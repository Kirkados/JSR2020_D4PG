# JSR2020_D4PG
Open-sourced deep guidance implementation using the D4PG algorithm for spacecraft proximity operations as detailed in Hovell and Ulrich's JSR 2020 paper under review titled "Deep Reinforcement Learning for Spacecraft Proximity Operations Guidance"

Built on Tensorflow 1.12.0

To run, optionally modify settings in environment_envs123456.py and settings.py. Then run python3 on main.py to begin training.

Deep reinforcement learning is used to train a neural network to output velocity commands for a spacecraft to track using a conventional controller. This "deep guidance" technique is a possible solution to the simulation-to-reality problem. The task-solving ability of deep reinforcement learning is harnessed along with the ability of conventional control to perform well under model uncertainty.

A video showing simulated and experimental results can be found here: https://youtu.be/n7K6aC5v0aY

Feel free to contact me if you have any questions! khovell@gmail.com
