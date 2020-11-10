#include "Gridworld.hpp"

using namespace std;
using namespace Eigen;

Gridworld::Gridworld(mt19937_64 & generator)
{
	// Make the uniform distribution be over the interval [0,1]
	d = uniform_real_distribution<double>(0, 1);

	// Prepare for the first episode
	newEpisode(generator);
}

int Gridworld::getMaxEps() const
{
	// You shouldn't need this many episodes with a good learning algorithm,
	// but we may not be greating very good algorithms in this assignment.
	return 300;
}

int Gridworld::getStateDim() const
{
	// One-hot encoding. We are not representing the terminal absorbing state (getState() will never be called when the agent is in the terminal absorbing state)
	return 23;
}

int Gridworld::getNumActions() const
{
	// Four discrete actions
	return 4;
}

double Gridworld::getGamma() const
{
	return 0.9;
}

double Gridworld::transition(const int& a, mt19937_64 & generator)
{
	// Check if we should transition to s_infty.
	if ((x == 4) && (y == 4))
	{
		TAS = true;
		return 0;
	}

	// Check if we should terminate due to running too long
	t++;
	if (t == 100)
	{
		TAS = true;
		return -100;
	}

	// We implement the "veer" and "stay" behavior with an "effective action" that is modified from the actual action "a"
	int effectiveAction = a;
	double temp = d(generator);		// Temp is a uniform random number from 0 to one.
	if (temp <= 0.1)				// This should pass 10% of the time
		effectiveAction = -1;		// Actions -1 causes the "stay" behavior below
	else if (temp <= 0.15)			// This should pass (but not the previous if-statement) 5% of the time.
	{
		effectiveAction++;			// Rotate the action 90 degrees
		if (effectiveAction == 4)
			effectiveAction = 0;	// Wrap around
	}
	else if (temp <= 0.2)			// This should happen 5% of the time
	{
		effectiveAction--;			// Rotate the action -90 degrees
		if (effectiveAction == -1)
			effectiveAction = 3;	// Wrap around
	}

	// Compute the resulting agent position
	int xPrime = x, yPrime = y;
	if ((effectiveAction == 0) && (y >= 1))
		yPrime--;		// Up
	else if (effectiveAction == 1)
		xPrime++;		// Right
	else if (effectiveAction == 2)
		yPrime++;		// Down
	else if (effectiveAction == 3)
		xPrime--;		// Left
	// Note that a == -1 is possible. We use this to implement the "stay" behavior. The above statement is NOT "else", it is "else if" to handle this -1 case.

	// If the new position is valid, then update to it!
	if ((xPrime >= 0) && (yPrime >= 0) && (xPrime < 5) && (yPrime < 5) &&	// Inside the grid
		((xPrime != 2) || ((yPrime != 2) && (yPrime != 3))))				// Not in an obstacle
	{
		x = xPrime;
		y = yPrime;
	}

	// Compute the resulting reward
	if ((x == 2) && (y == 4))
		return -10;					// The agent is in the water state
	else if ((x == 4) && (y == 4))
		return 10;					// The agent is in the bottom-right "goal" state.
	else
		return 0;					// The agent isn't in the water or the "goal" state, so the reward is zero
}

VectorXd Gridworld::getState() const
{
	// Create the object we will return, and initialize to the zero-vector, of length 23.
	VectorXd result = VectorXd::Zero(23);
	if (!TAS)	// If we are in the terminal absorbing state, this shouldn't be called. Just in case, let's use the all-zero vector to denote the TAS
	{
		int state = y * 5 + x;	// map the x,y coordinates to a number in [0,24]

		// Cut the two obstacles. You can work out with pen and paper that this should do what we want. Or, you could run the "manual" agent and have the agent walk around the environment to confirm the desired behavior.
		assert(state != 12);	// This is the upper obstacle. Note that states start at 0 here, unlike the course notes where they start at 1
		if (state > 12)
			state--;
		assert(state != 16);	// This is the lower obstacle (after being shifted left one)
		if (state > 16)
			state--;
		result[state] = 1;
	}
	
	// Return the computed state representation
	return result;
}

bool Gridworld::inTAS() const
{
	return TAS;
}

void Gridworld::newEpisode(mt19937_64 & generator)
{
	// Start at position (0,0), with the time counter also equal to zero
	x = y = t = 0;

	// We are not starting in the terminal absorbing state
	TAS = false;
}

