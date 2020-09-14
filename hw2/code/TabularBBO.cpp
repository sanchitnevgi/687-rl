#include "TabularBBO.hpp"

using namespace std;				// So we don't have to keep writing std::
using namespace Eigen;				// So we don't have to keep writing Eigen::

// See the .hpp file for a description of this "constructor"
// Note: If you want to use a different value for N, pass that value at the end of the line below, e.g., EpisodicAgent(2)
TabularBBO::TabularBBO(const int& stateDim, const int& numActions, const double& gamma, const int& N, const int & maxEps) : EpisodicAgent(N)
{
	// @TODO: You can add code here, but do not change the argument list above.
}

// Ask the agent to select an action given the current state
int TabularBBO::getAction(const Eigen::VectorXd& s, std::mt19937_64& generator) const
{
	// @TODO: Write this function
	return uniform_int_distribution<int>(0,3)(generator);	// Do not keep this line - for now it implements the uniform random policy.
}

// Reset the agent entirely - to a blank slate prior to learning
void TabularBBO::reset(std::mt19937_64& generator)
{
	// @TODO: Write this function
}

void TabularBBO::episodicUpdate(mt19937_64& generator)
{
	// @TODO: Write this function
}