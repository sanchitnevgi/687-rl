#include "QLearning.hpp"

using namespace std;
using namespace Eigen;

/*
* In this assignment, the constructor takes the environment name and
* uses it to set hyperparameters like the step size, gamma, and any other parameters.
*/
QLearning::QLearning(const int& stateDim, const int& numActions, const std::string& envName)
{
	// @TODO: Fill in this function

	// In the constructor, use envName to set hyperparameters.
	// No not use envName to initialize the policy!
	// I'm leaving in some of my code to show how you might structure this.
	if (envName.compare("Mountain Car") == 0)
	{
		// @TODO: Set hyperparameters for Mountain Car.
	}
	else if (envName.compare("Cart Pole") == 0)
	{
		// @TODO: Set hyperparameters for Cart Pole.
	}
	else if (envName.compare("Gridworld") == 0)
	{
		// @TODO: Set hyperparameters for Gridworld.
	}
	else
	{
		cout << "Error: Unknown environment name in Sarsa constructor." << endl;
		exit(1);
	}

	// @TODO: Fill in the remainder of this function.
}

bool QLearning::updateBeforeNextAction()
{
	return true;
}

// Epsilon-greedy or Softmax action selection
int QLearning::getAction(const Eigen::VectorXd& s, std::mt19937_64& generator)
{
	// @TODO: Fill in this function.
	return 0; // Delete this line - this is just here to that the code compiles.
}

// Tell the agent that it is at the start of a new episode
void QLearning::newEpisode()
{
	// @TODO: Fill in this function.
}

// Update given a (s,a,r,s') tuple
void QLearning::update(const Eigen::VectorXd& s, const int& a, const double& r, const Eigen::VectorXd& sPrime, std::mt19937_64& generator)
{
	// @TODO: Fill in this function.
}

// Let the agent update/learn when sPrime would be the terminal absorbing state
void QLearning::update(const Eigen::VectorXd& s, const int& a, const double& r, mt19937_64& generator)
{
	// @TODO: Fill in this function.
}

/*
* My code used this function. It returns a^b, where a and b are both integers.
* You may not need this, but if you do, we are including it.
*/
int QLearning::ipow(const int& a, const int& b) {
	if (b == 0) return 1;
	if (b == 1) return a;
	int tmp = ipow(a, b / 2);
	if (b % 2 == 0) return tmp * tmp;
	else return a * tmp * tmp;
}

/*
* This is another function that my code uses, and which you are welcome to use.
* It is entirely possible that your code won't require this function.
* This function takes as input a vector, buff, that represents a number in
* base (maxDigit+1), and adds one to the counter. Overflow behavior is not defined.
*/
void QLearning::incrementCounter(VectorXd& buff, const int& maxDigit) {
	for (int i = 0; i < (int)buff.size(); i++) {
		buff[i]++;
		if (buff[i] <= maxDigit)
			break;
		buff[i] = 0;
	}
}