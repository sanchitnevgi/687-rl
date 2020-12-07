#include "TabularBBO.hpp"	// The .cpp file for an class should include the header file (the .hpp file).

using namespace std;				// So we don't have to keep writing std::
using namespace Eigen;				// So we don't have to keep writing Eigen::

// See the .hpp file for a description of this "constructor"
TabularBBO::TabularBBO(const int& stateDim, const int& numActions, const double& gamma, const int& N, const int& maxEps) : EpisodicAgent(N)
{
	this->numStates = stateDim;									// Remember the number of states in a private member variable
	this->numActions = numActions;								// Remember the number of actions in a private member variable
	this->maxEps = maxEps;
	this->gamma = gamma;										// Remember gamma too
	curTheta = newTheta = MatrixXd::Zero(numStates, numActions);// Initialize both the current and new theta to the zero Matrix
	curThetaJHat = -INFINITY;									// Set the current policy's performance to -INFINITY so that we always take the first policy tested.
	epCount = 0;
}

// Ask the agent to select an action given the current state
int TabularBBO::getAction(const Eigen::VectorXd& s, std::mt19937_64& generator) const
{
	// Convert the one-hot state into an integer from 0 - (numStates-1)
	int state;
	for (state = 0; state < (int)s.size(); state++)
		if (s[state] != 0)
			break;
	assert(state != (int)s.size());					// If this happens, the s-vector was all zeros

	// Get the action probabilities from theta, using softmax action selection.
	VectorXd actionProbabilities = newTheta.row(state).array().exp();
	actionProbabilities.array() /= actionProbabilities.sum();

	// Sample an action from actionProbabilities.
	// The <double> below means that the object "uniform_real_distribution" is a "templated" object. That means
	// that the object itself is defined for a specific type. In this case, we are creating a "double" version of the object.
	// Here, you could use <float> to get a single-precision floating point generator rather than a double precision.
	// using #include <random> we get many different distributions. uniform_real_distribution<double> is one. Its constructor
	// takes the bounds of the uniform real distribution, in this case [0,1]. Below, "uniform_real_distribution<double>(0, 1)" creates
	// an object calling the constructor with (0,1), and then we immediately use it with (generator), which samples the distribution
	// using the provided random number generator
	//
	// A more common form is:
	// >>> uniform_real_distribution<double> myDistribution(0,1);
	// Then, you can sample this distribution at any time with:
	// >>> myDistribution(generator);
	// where "generator" is an object of type mt19937_64 (for example).
	//
	double temp = uniform_real_distribution<double>(0, 1)(generator), sum = 0;
	for (int a = 0; a < numActions; a++)
	{
		sum += actionProbabilities[a];
		if (temp <= sum)
			return a;	// The function will return 'a'. This stops the for loop and returns from the function.
	}
	return numActions - 1; // Rounding error
}

// Reset the agent entirely - to a blank slate prior to learning
void TabularBBO::reset(std::mt19937_64& generator)
{
	// You can chain together equal statements like this:
	curTheta = newTheta = MatrixXd::Zero(numStates, numActions);
	curThetaJHat = -INFINITY;
	epCount = 0;
	EpisodicAgent::reset(generator);	// See EpisodicAgent::reset. Here TabularBBO is the subclass and EpisodicAgent is the superclass. EpisodicAgent has its own variables to reset, but calling "reset" calls the function for the subclass. So, this line is saying "also call the reset function for EpisodicAgent too!"
}

void TabularBBO::episodicUpdate(mt19937_64& generator)
{
	// Track how many episodes have passed
	epCount += N;

	// We are going to compute newThetaJHat (an estimate of how good the new policy is), and will then
	// see if it is better than the best policy we found so far.
	newThetaJHat = 0;

	// Loop over the N episodes
	for (int epCount = 0; epCount < N; epCount++)
	{
		// Compute the return
		double curGamma = 1;
		int epLen = (int)rewards[epCount].size();
		for (int t = 0; t < epLen; t++)
		{
			newThetaJHat += curGamma * rewards[epCount][t];
			curGamma *= gamma;
		}
	}
	newThetaJHat /= (double)N;

	// Is the new policy better than our current best?
	if (newThetaJHat > curThetaJHat)
	{
		// It looks like it! Change our current best
		curTheta = newTheta;
		curThetaJHat = newThetaJHat;
	}

	// If we randomly sample policies forever, the average performance won't go up.
	// For the last 10% of episodes, we'll just run the best policy we found so far.
	uniform_real_distribution<double> d(-2, 2);
	for (int s = 0; s < numStates; s++)
		for (int a = 0; a < numActions; a++)
			newTheta(s, a) = curTheta(s,a) + d(generator);
}