---
title: "Chainlink Home Assignment"
layout: post
author: Olga Kahn
projects: true
category: project
summary:
permalink: projects/chainlink
---

# Python vs Power BI for a Jr Data Analyst takehome assignment 

![png](/assets/images/posts/chainlink/chainlink_main.png)

As part of a job application for a junior data analyst position at a blockchain company I was tasked to do an automated dashboard looking at the Total Value Locked, or Total Value Secured, of their clients vs the entire entire DeFi market:
![png](/assets/images/posts/chainlink/chainlink_task1.png)
![png](/assets/images/posts/chainlink/chainlink_task2.png)

I first did this using python (with pandas) and jupyter notebook (and later [hosted it on heroku](https://chainlink-tvl.herokuapp.com/)). I then also did the same process using Microsoft’s Power BI tool and its DAX language, which I never used before. Both versions are available in [this repo](https://github.com/olgadk7/chainlink_tvs). To use PowerBI, a licensed microsoft product, I downloaded [Parallels](https://www.parallels.com/​​) free trial of a windows virtual machine on my mac, and PowerBI came with it. 

Both versions go through the following 3 steps:

**1. Get assignment’s constraints from the google sheet**

For my Python version I wrote a function that uses pandas read_csv with a url, followed by a couple of transformations. 
Getting data in Power BI is done via creating queries in the “Power Query” editor. Click Get Data > Web and input the link in a certain way, as per [this advice](https://www.bizone.co.th/blogs/business-intelligence/part-3-connecting-power-bi-google-sheets). Do the standard transformations, close & apply. 

**2. Pull the protocols’ TVL data from DeFi Llama API**

The Python version uses the requests library, calling their get function on every protocol in the lists “users” and “non-users” pulled from the google sheet, and turning it into json, which is then flattened with pandas’ json_normalize.
The Power BI version uses the Power Query’s Get Data via Web functionality again. Once a single a single query worked on one protocol, we can make it iterate over multiple url inputs (i.e. protocols), by creating a protocol “parameter”, creating a function and then invoking it on the column of those protocols, as described [here](https://wisedatadecisions.com/2021/05/03/parameterize-an-api-request-in-power-query). Note, I tried a similar method, but it made the Power BI “unresponsive” no matter what I tried. 

**3. Plotting.**

For the python version I used Plotly for its interactivity as well as compatibility with Dash and Heroku. 
Visualization in PowerBI is really straightforward, mostly clicking, dragging and dropping. The only thing that’s recommended to do that wasn’t intuitive is to make a separate calendar table that would ensure that every date is present in the graph and thereby the time axis is continuous. Also make sure the date is of “date” format and sorted as such. 

**Things that took me a lot of time to get used to in PowerBI**
1. Referencing between Power Query and its three view modes (“report”, “data”, and “model”). If you change something in place, the flow might break in another place. 
2. Power BI applies automatic transformations on data from the web that are governed in the applied steps as well as DAX editor, and these 2 have to correspond.
3. Don’t forget to manually save your work..
