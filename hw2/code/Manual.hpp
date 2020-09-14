#pragma once

#include "Agent.hpp"

/*
* This agent lets you enter actions via the console.
*/
class Manual : public Agent
{
public:
	// Should the agent update after getting (s,a,r,s') or after getting (s,a,r,s',a')?
	static bool updateBeforeNextAction();

	// Ask the agent to select an action given the current state
	int getAction(const Eigen::VectorXd& s, std::mt19937_64& generator) const override;

	// Tell the agent that it is at the start of a new episode
	void newEpisode() override;

	// Reset the agent entirely - to a blank slate prior to learning
	void reset(std::mt19937_64& generator) override;

	// Update given the transition
	void update(const Eigen::VectorXd& s, const int& a, const double& r, const Eigen::VectorXd& sPrime, std::mt19937_64& generator) override;
	
	// Let the agent update/learn when sPrime would be the terminal absorbing state
	void update(const Eigen::VectorXd& s, const int& a, const double& r, std::mt19937_64& generator) override;
};