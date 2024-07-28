# Ride-Hailing Trip Assignment Analysis

This Jupyter notebook analyzes the trip assignment method used by a ride-hailing app. The app currently assigns new incoming trips to the closest available vehicle using the Haversine distance, referred to as the linear distance. The [Haversine formula](https://en.wikipedia.org/wiki/Haversine_formula) calculates the shortest distance over the Earth's surface between two points. However, this method does not accurately reflect actual travel times in urban environments, leading to potential inefficiencies in trip assignments.

## Introduction
In urban settings, the expected travel time from point A to point B is not solely defined by the linear (Haversine) distance. Cities are complex networks with extensive transportation infrastructure, including roads, highways, bridges, and tunnels, designed to increase capacity and reduce travel times. This infrastructure means that the bird distance (linear distance) often fails as a reliable proxy for estimating travel time.

To address this, the concept of road distance is introduced, which calculates the shortest path a vehicle would take using the road network. Road distance accounts for the actual routes vehicles must take, including turns, intersections, and specific road conditions, providing a more realistic estimate of travel time between two points.

This discrepancy between linear and road distances is evident in urban areas like Mexico City (CDMX). In the image below, the blue area indicates regions reachable within a 10-minute drive, demonstrating how actual travel times deviate from the simplistic linear distance model.

<p align="center">
  <img src="![Mexico_DF.png](Mexico_DF.png)"
       alt="CDMX Isochrone Example">
</p>

## Purpose
The purpose of this analysis is to compare the effectiveness of using linear distance versus road distance for trip assignments. By understanding the differences and their impact on travel times, we can improve the efficiency of the ride-hailing app's trip assignment algorithm.

