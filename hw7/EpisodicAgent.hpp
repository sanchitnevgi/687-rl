#pragma once

#include "Agent.hpp"

/*
* Create a subclass of this class for your blackbox optimization algorithms. The EpisodicAgent object itself
* handles running episodes, and called episodicUpdate() whenever an episode is completed. This makes it easier to
* write algorithms like BBO algorithms, which don't update every timestep.
* 
* Any subclass should implement the functions "getAction", "episodicUpdate", and "reset". The reset function should end with "EpisodicAgent::reset()" to call this parent's class reset function.
*/
class EpisodicAgent : public Agent
{
public:
	EpisodicAgent(const int & N);					// The "episodicUpdate" function will be called after every N episodes
	
	static bool updateBeforeNextAction();

	// Ask the agent to select an action given the current state
	virtual int getAction(const Eigen::VectorXd& s, std::mt19937_64& generator) const override = 0;

	// Tell the agent that it is at the start of a new episode
	void newEpisode() override;

	// Reset the agent entirely - to a blank slate prior to learning
	void reset(std::mt19937_64& generator) override = 0;

	// Update given a (s,a,r,s') tuple
	void update(const Eigen::VectorXd& s, const int& a, const double& r, const Eigen::VectorXd& sPrime, std::mt19937_64& generator) override;
	
	// Let the agent update/learn when sPrime would be the terminal absorbing state
	void update(const Eigen::VectorXd& s, const int& a, const double& r, std::mt19937_64& generator) override;

	virtual void episodicUpdate(std::mt19937_64& generator) = 0;

protected:
	int N;
	int epCount;
	std::vector<std::vector<Eigen::VectorXd>> states;
	std::vector<std::vector<int>> actions;
	std::vector<std::vector<double>> rewards;

	void wipeStatesActionsRewards();

};
