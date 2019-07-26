PEW RESEARCH CENTER FOR THE PEOPLE & THE PRESS
MARCH POLITICAL
March 7-14, 2018
N=1,466

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

Releases from this survey:

March 15, 2018. "Public Confidence in Mueller’s Investigation Remains Steady"
http://www.people-press.org/2018/03/15/public-confidence-in-muellers-investigation-remains-steady/

March 22, 2018. "Positive Views of Economy Surge, Driven by Major Shifts Among Republicans"
http://www.people-press.org/2018/03/22/positive-views-of-economy-surge-driven-by-major-shifts-among-republicans/

April 26, 2018. "The Public, the Political System and American Democracy"
http://www.people-press.org/2018/04/26/the-public-the-political-system-and-american-democracy/



Blog posts from this survey:

March 15, 2018. "Disagreements about Trump widely seen as reflecting divides over ‘other values and goals’"
http://www.pewresearch.org/fact-tank/2018/03/15/disagreements-about-trump-widely-seen-as-reflecting-divides-over-other-values-and-goals/

March 19, 2018. "The Iraq War continues to divide the U.S. public, 15 years after it began"
http://www.pewresearch.org/fact-tank/2018/03/19/iraq-war-continues-to-divide-u-s-public-15-years-after-it-began/

***************************************************************************************************************************

SYNTAX

***The following syntax is for constructed demographic variables***.

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