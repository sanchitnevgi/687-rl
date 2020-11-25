#pragma once

#include "Environment.hpp"

class MountainCar : public Environment
{
public:
	MountainCar(std::mt19937_64 & generator);

	int getMaxEps() const override;
	int getStateDim() const override;
	int getNumActions() const override;
	double getGamma() const override;
	double transition(const int& a, std::mt19937_64& generator) override;
	Eigen::VectorXd getState() const override;
	bool inTAS() const override;
	void newEpisode(std::mt19937_64& generator) override;

private:
	const double minX = -1.2;
	const double maxX = 0.5;
	const double minXDot = -0.07;
	const double maxXDot = 0.07;

	int t;

	Eigen::VectorXd state;	// [2] - x and xDot

	static double bound(const double& x, const double& minValue, const double& maxValue);
	static double normalize(const double& x, const double& minValue, const double& maxValue);
};