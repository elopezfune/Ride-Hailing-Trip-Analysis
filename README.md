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


## Experimental design
The designed expirement is very simple. For a period of 5 days, all trips in 3 cities (Bravos, Pentos and Volantis) have been randomly assigned using linear or road distance:
1. Trips whose trip_id starts with digits 0-8 were assigned using road distance
2. Trips whose trip_id starts with digits 9-f were assigned using linear distance


## Data description
The collected data is available available in the data directory. Each object represent a vehicle_interval that contains the following attributes:
    type: can be going_to_pickup, waiting_for_rider or driving_to_destination
    trip_id: uniquely identifies the trip
    duration: how long the interval last, in seconds
    distance: how far the vehicle moved in this interval, in meters
    city_id: either bravos, pentos and volantis
    started_at: when the interval started, UTC Time
    vehicle_id: uniquely identifies the vehicle
    rider_id: uniquely identifies the rider
    
    
### Example
{<br>
 "duration": 857,<br>
 "distance": 5384,<br>
 "started_at": 1475499600.287,<br>
 "trip_id": "c00cee6963e0dc66e50e271239426914",<br>
 "vehicle_id": "52d38cf1a3240d5cbdcf730f2d9a47d6",<br>
 "city_id": "pentos",<br>
 "type": "driving_to_destination"<br>
 }

## Purpose
The purpose of this analysis is to compare the effectiveness of using linear distance versus road distance for trip assignments. By understanding the differences and their impact on travel times, we can improve the efficiency of the ride-hailing app's trip assignment algorithm.


## Challenge
Try to answer the following questions:
1. Should the company move towards road distance? What's the max price it would make sense to pay per query? (make all the assumptions you need, and make them explicit)
2. How would you improve the experimental design? Would you collect any additional data?

