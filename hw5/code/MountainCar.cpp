#include "MountainCar.hpp"

using namespace std;
using namespace Eigen;

MountainCar::MountainCar(mt19937_64 & generator)
{
	state.resize(2);
	newEpisode(generator);
}

int MountainCar::getMaxEps() const
{
	return 500;
}

int MountainCar::getStateDim() const
{
	return 2;
}

int MountainCar::getNumActions() const
{
	return 3;
}

double MountainCar::getGamma() const
{
	return 1.0;
}

double MountainCar::transition(const int& a, mt19937_64 & generator)
{
	t++;
	double u = (double)a - 1.0;	// Convert act to a double in {-1, 0, 1}
	// Update xDot and then x
	state[1] = bound(state[1] + 0.001 * u - 0.0025 * cos(3.0 * state[0]), minXDot, maxXDot);
	state[0] += state[1];
	if (state[0] < minX) {
		state[0] = minX;
		state[1] = 0;					// Inelastic collisions
	}
	if (state[0] > maxX)
		state[0] = maxX;
	return -1;							// Reward is always -1
}

VectorXd MountainCar::getState() const
{
	VectorXd result(2);
	result[0] = normalize(state[0], minX, maxX);
	result[1] = normalize(state[1], minXDot, maxXDot);
	return result;
}

bool MountainCar::inTAS() const
{
	return (state(0) >= maxX) || (t >= 20000);
}

void MountainCar::newEpisode(mt19937_64 & generator)
{
	t = 0;
	state[0] = -0.5;
	state[1] = 0;
}

double MountainCar::bound(const double& x, const double& minValue, const double& maxValue)
{
	return min(maxValue, max(minValue, x));
}

double MountainCar::normalize(const double& x, const double& minValue, const double& maxValue)
{
	double temp = bound(x, minValue, maxValue);
	return (x - minValue) / (maxValue - minValue);
}

