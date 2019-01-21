---
title: "Landlord issues: using NYC OpenData to explore housing safety & rent stabilization"
layout: post
blog: true
author: Olga Krieger
summary:
permalink: blog/NYCOpenData
---
# Landlord issues: using NYC OpenData to explore housing safety & rent stabilization

Public infrastructure is one of the things I love about New York: parks, libraries, classes and, as I’m discovering, data. Recently I took an OpenData class hosted by [BetaNYC](https://beta.nyc/), which has designed tools intended to help Community Boards access, analyze, map and use data to inform decision-making. Below is a project that came out of one of those classes. Using tools such as BoardStat, Tenants Map, BIS-WEB, ACRIS and RentLogic, as well as 311 data, the study explores the connection between housing safety issues and rent-regulated units. As New York City is struggling through an [affordable housing crisis](https://nyti.ms/2GxIkF7), the study, not surprisingly, finds a connection between bad landlords, housing violations and rent stabilization.

<br><br>

## Part I: Identify a building with the most Heat / Hot Water service requests.
Having chosen Manhattan as a borough of interest on [BoardStat](https://betanyc.github.io/BoardStat/), I further filtered the data selecting Community Board 02 Manhattan. It gives us an overview of the top complaint types and addresses for that community board. We’re going to focus on the Heat / Hot Water complaint in entire buildings as it’s an essential need in the cold months. The address with the most complaints of this type is 1 University Place:

![png](/assets/images/posts/lanlord-issues/1.png)

![png](/assets/images/posts/lanlord-issues/2.png)

<br><br><br>

## Part II: Discovering other complaint types and the trend line for Heat / Hot Water complaints. 
The top complaints at 1 University Place are Heat / Hot Water and different types of noises: 

![png](/assets/images/posts/lanlord-issues/3.png)

<br>

Zooming in on the heat / hot water, we see that the number of complaints peaks in 2018, almost tripling compared to the previous year:

![png](/assets/images/posts/lanlord-issues/4.png)

<br><br><br>

## Part III: Determining whether the property is rent-regulated.
We use Tenants Map to determine whether 1 University Place is rent-regulated and indeed it is: the lot is outlined in black solid line, which means that it contains rent-stabilized units according to both the Rent Guidelines Board and NYC Property Tax Bills, and it’s shaded orange, depicting the increasing number of housing-related 311 complaints made about the property since 2010. 

![png](/assets/images/posts/lanlord-issues/5.png)

Clicking on the lot, a summary of the building info comes up.

We see that the amount of rent-stabilized units has decreased by 33% between 2007 and 2016:

![png](/assets/images/posts/lanlord-issues/6.png)

Looking at the 311 complaints, we see that the heat / hot water is by far the most overwhelming complaint, 85.5% of the total. 

![png](/assets/images/posts/lanlord-issues/7.png)

Total Number of Complaints: 159

The number of complaints has has risen in recent years, more than doubling in 2018 compared to the previous year. 

![png](/assets/images/posts/lanlord-issues/8.png)

Source: [311 Service Requests from 2010 to Present](https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2010-to-Present/erm2-nwe9/data), [Rent Guidelines Board Rent Stabilized Building Lists](https://www1.nyc.gov/site/rentguidelinesboard/resources/rent-stabilized-building-lists.page), [Department of Finance Property Tax Bills](https://webapps.nyc.gov/CICS/fin1/find001i)
 
<br><br><br>

## Part IV. Identifying buildings with the most Housing Preservation & Development (HPD) service requests.
Going back to BoardStat (page 3), we’re going to filter by the agency, selecting HPD from the drop-down menu. HPD is the agency responsible for maintaining the city's affordable housing; the larger the circle, the higher the density of HPD requests, signifying unsafe housing conditions. In 2018, the largest circle is 46 Bank St.

![png](/assets/images/posts/lanlord-issues/9.png)

Locating the address on the Tenants Map, we confirm that the lot is rent-stabilized, with a high number of housing-related 311 complaints. The summary tells us that the owner is 46 ROSE REALTY, LLC. 

![png](/assets/images/posts/lanlord-issues/10.png)

![png](/assets/images/posts/lanlord-issues/11.png)

<br>

Rent-Stabilized Units

![png](/assets/images/posts/lanlord-issues/12.png)
 
311 Complaints

![png](/assets/images/posts/lanlord-issues/13.png)

Total Number of Complaints: 324

![png](/assets/images/posts/lanlord-issues/14.png)

Source: [311 Service Requests from 2010 to Present](https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2010-to-Present/erm2-nwe9/data), [Rent Guidelines Board Rent Stabilized Building Lists](https://www1.nyc.gov/site/rentguidelinesboard/resources/rent-stabilized-building-lists.page), [Department of Finance Property Tax Bills](https://webapps.nyc.gov/CICS/fin1/find001i)
 
<br><br><br>

## Part V. Noting violations and transaction history for a problematic address. 
Going over to [NYCityMap](http://maps.nyc.gov/doitt/nycitymap/) and clicking Building Profile under Links to More Information, we see right away that Class 1: Immediately Hazardous violation has been issued at 46 Bank st. In addition, there are 96 complaints, 20 Department of Buildings violations (4 of which are open) and 26 Environmental Control Board violations (21 of which are open):

![png](/assets/images/posts/lanlord-issues/15.png)

Clicking on Tax and Property Records (also under Links to More Information), we get to a Department of Finance tool, ACRIS (Automated City Register Information System). The current owner, 46 ROSE REALTY LLC, gained the title in Spring 2017: 

![png](/assets/images/posts/lanlord-issues/16.png)

<br><br><br>

## Part VI. Using RentLogic to see the building’s rating / letter grade. 
[RentLogic](https://rentlogic.com) is a company which rates building on a scale from A - F by consolidating and weighing a large number of open datasets, using an open, 250-page algorithm. Having been graded an A from the beginning of the data collection through the summer of 2017, 46 Bank St, dropped to an F in just half a year. The decline happened a the same time as 46 ROSE REALTY LLC took over the building. Among the listed violations are: Water, Plumbing, Safety, Unsanitary Conditions, Fire Code, Cockroaches, Rodents, Electrical, Gas, Hot Water and Heat.

![png](/assets/images/posts/lanlord-issues/17.png)
