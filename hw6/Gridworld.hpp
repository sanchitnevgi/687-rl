#pragma once

#include "Environment.hpp"

/*
* 687-Gridworld! [Note: Modified to terminate episodes after 100 time steps]
* 
* We view the top left (start) at [0,0], and store states in x,y coordinates internally, even though
* we map these to states {1,2,...,23} when returning the state.
* 
* The actions are:
* 0: Up
* 1: Right
* 2: Down
* 3: Left
*/

class Gridworld : public Environment
{
public:
	Gridworld(std::mt19937_64 & generator);

	int getMaxEps() const override;
	int getStateDim() const override;
	int getNumActions() const override;
	double getGamma() const override;
	double transition(const int& a, std::mt19937_64& generator) override;
	Eigen::VectorXd getState() const override;
	bool inTAS() const override;
	void newEpisode(std::mt19937_64& generator) override;

private:
	int x;			// Agent horizontal coordinate (0 to 4)
	int y;			// Agent vertical coordinate (0 to 4)
	int t;			// Time into the episode. We terminate after 100 time steps
	bool TAS;		// Are we in the terminal absorbing state?
	std::uniform_real_distribution<double> d;	// A uniform random distribution we will use a few times.
};