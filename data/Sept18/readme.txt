PEW RESEARCH CENTER FOR THE PEOPLE & THE PRESS
SEPTEMBER POLITICAL
September 18-24, 2018*
N=1,754

***************************************************************************************************************************

This dataset includes cell phone interviews conducted using an RDD sample of cell phone numbers. 
Cell phone interviews include households that are cell-only as well as those that also have a landline phone. 
The dataset contains several weight variables. 

WEIGHT is the weight for the combined sample of all landline and cell phone interviews. 
Data for all Pew Research Center reports are analyzed using this weight.

One additional weight can be used to compare the combined sample with the cell phone sample by itself.

CELLWEIGHT is for analysis of the cell RDD sample only. Interviews conducted by landline are not
given a weight and are excluded from analysis when this weight is used.

***************************************************************************************************************************

Beginning in the Fall of 2008, the Pew Research Center started using respondents’ self-reported zip code as  
the basis for geographic variables such as region, state and county. We do this because the error rate in the original 
geographic information associated with the sample is quite high, especially for respondents from the cell phone sample. 

For respondents who do not provide a zip code or for those we cannot match, we use the sample geographic information. 
We continue to include the original sample geographic variables in the datasets (these variables are preceded by an ‘s’) 
for archival purposes only.

To protect the privacy of respondents, telephone numbers, county of residence and zip code have been removed from the data file.

***************************************************************************************************************************

*Three questions about the Russia investigation and Robert Mueller (Q93, Q96, Q97) were pulled from the field early (September 23); the sample size for these three questions is N=1,622. 

***************************************************************************************************************************


Releases from this survey:

September 26, 2018. "Voter Enthusiasm at Record High in Nationalized Midterm Environment"
http://www.people-press.org/2018/09/26/voter-enthusiasm-at-record-high-in-nationalized-midterm-environment/

October 1, 2018. "Trump Gets Negative Ratings for Many Personal Traits, but Most Say He Stands Up for His Beliefs." 
http://www.people-press.org/2018/10/01/trump-gets-negative-ratings-for-many-personal-traits-but-most-say-he-stands-up-for-his-beliefs/

October 4, 2018. "2018 Midterm Voters: Issues and Political Values." 
http://www.people-press.org/2018/10/04/2018-midterm-voters-issues-and-political-values/

October 18, 2018. "Gun Policy Remains Divisive, But Several Proposals Still Draw Bipartisan Support." 
http://www.people-press.org/2018/10/18/gun-policy-remains-divisive-but-several-proposals-still-draw-bipartisan-support/


Blog posts from this survey:

September 24, 2018. "Views of Mueller's investigation -- and Trump's handling of the probe -- turn more partisan." 
http://www.pewresearch.org/fact-tank/2018/09/24/views-of-muellers-investigation-and-trumps-handling-of-the-probe-turn-more-partisan/#more-306310

September 28, 2018. "10 years after the financial crisis, Americans are divided on security of U.S. economic system." 
http://www.pewresearch.org/fact-tank/2018/09/28/americans-are-divided-on-security-of-u-s-economic-system/ 

October 3, 2018. "Most continue to say ensuring health care coverage is government's responsibility."
http://www.pewresearch.org/fact-tank/2018/10/03/most-continue-to-say-ensuring-health-care-coverage-is-governments-responsibility/

October 4, 2018. "Partisans are divided over the fairness of the U.S. economy -- and why people are rich or poor." 
http://www.pewresearch.org/fact-tank/2018/10/04/partisans-are-divided-over-the-fairness-of-the-u-s-economy-and-why-people-are-rich-or-poor/

October 5, 2018. "After 17 years of war in Afghanistan, more say U.S. has failed than succeeded in achieving its goals." 
http://www.pewresearch.org/fact-tank/2018/10/05/after-17-years-of-war-in-afghanistan-more-say-u-s-has-failed-than-succeeded-in-achieving-its-goals/

October 8, 2018. "About six-in-ten Americans support marijuana legalization." 
http://www.pewresearch.org/fact-tank/2018/10/08/americans-support-marijuana-legalization/

October 9, 2018. "Most Americans view openness to foreigners as 'essential to who we are as a nation.' 
http://www.pewresearch.org/fact-tank/2018/10/09/most-americans-view-openness-to-foreigners-as-essential-to-who-we-are-as-a-nation/

October 17, 2018. "Nearly six-in-ten Americans say abortion should be legal in all or most cases." 
http://www.pewresearch.org/fact-tank/2018/10/17/nearly-six-in-ten-americans-say-abortion-should-be-legal/ 


*************************************************************************************************************************
SYNTAX 

***The following syntax is for constructed demographic variables*** 

*The combined race variable (racecmb) was computed using the following syntax:
recode race_1 (1=1) (2=2) (3=3) (4 thru 7=5) (8 thru 9=9) into racecmb.
if race_2>0 and race_2 <8 racecmb=4.
variable label racecmb "Combining Race".
value label racecmb
1 "White"
2 "Black or African-American"
3 "Asian or Asian-American"
4 "Mixed Race"
5 "Or some other race"
9 "Don’t know/Refused (VOL.)".

*The race-ethnicity variable (racethn) was computed using the following syntax:
if racecmb=1 and hisp ge 2 racethn=1.
if racecmb=2 and hisp ge 2 racethn=2.
if (racecmb ge 3 and racecmb le 5) and (hisp ge 2) racethn=4.
if racecmb=9 racethn=9.
if hisp=1 racethn=3.
variable label racethn “Race-Ethnicity”.
value label racethn
1 “White non-Hispanic”
2 “Black non-Hispanic”
3 “Hispanic”
4 “Other”
9 “Don’t know/Refused (VOL.)”.


*The questions asked of Catholics specifically (q100a-q100d) were asked of all Catholics. Four respondents answered that they were 'Something else' but after an open-ended follow up, specified they were Catholics. They were not included in the Catholic-specific questions and marked as 'undesignated' using the following syntax: 

weight off.
compute cathundesig=0.
if respid=200264
or respid=200297
or respid=201005
or respid=201466 cathundesig=1.
fre cathundesig.

rename variables (q100a=q100aorig) (q100b=q100borig) (q100c=q100corig) (q100d=q100dorig).

compute q100a=q100aorig.
compute q100b=q100borig.
compute q100c=q100corig.
compute q100d=q100dorig.

if cathundesig=1 q100a=8888.
if cathundesig=1 q100b=8888.
if cathundesig=1 q100c=8888.
if cathundesig=1 q100d=8888.
execute.

apply dictionary from *
  /source variables = q100aorig 
  /target variables = q100a 
  /newvars.

add val label q100a 8888 'undesignated'.

apply dictionary from *
  /source variables = q100borig 
  /target variables = q100b 
  /newvars.

add val label q100b 8888 'undesignated'.

apply dictionary from *
  /source variables = q100corig 
  /target variables = q100c 
  /newvars.

add val label q100c 8888 'undesignated'.

apply dictionary from *
  /source variables = q100dorig 
  /target variables = q100d 
  /newvars.
add val label q100d 8888 'undesignated'.
execute.



