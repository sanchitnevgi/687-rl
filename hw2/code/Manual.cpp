#include <iostream>
#include "Manual.hpp"

using namespace std;
using namespace Eigen;

// Should the agent update after getting (s,a,r,s') or after getting (s,a,r,s',a')?
// It doesn't really matter for this agent, other than indicating which update function we
// need to override.
bool Manual::updateBeforeNextAction()
{
	return true;
}

// Ask the agent to select an action given the current state
int Manual::getAction(const VectorXd& s, mt19937_64& generator) const
{
	// If you want more practice, change the below to print out the state as an integer matching the 
	// course notes rather than as a one-hot vector. When testing this code, that's actually what I did.
	// We're using the one-hot encoding because it won't be specific to 687-Gridworld.
	int result;
	cout << "Reminder: 0 = Up, 1 = Right, 2 = Down, 3 = Left\nEnter action for state [" << s.transpose() << "]: ";
	cin >> result;	// This is how we read from the console: "console in"
	assert((result >= 0) && (result <= 3));	// Bounds-check the user's input
	return result;
}

// Tell the agent that it is at the start of a new episode
void Manual::newEpisode()
{
	cout << "Starting a new episode!" << endl;
}

void Manual::reset(mt19937_64& generator)
{
	cout << "You have been reset. Forget everything you learned (but not anything for CMPSCI 687!)." << endl;
}

// Update given the transition
void Manual::update(const VectorXd& s, const int& a, const double& r, const VectorXd& sPrime, mt19937_64 & generator)
{
	cout << "You moved to state [" << s.transpose() << "] and received a reward of " << r << endl;
}

// Let the agent update/learn when sPrime would be the terminal absorbing state
void Manual::update(const VectorXd& s, const int& a, const double& r, mt19937_64& generator)
{
	cout << "You moved to the terminal absorbing state and received a reward of " << r << endl;
}