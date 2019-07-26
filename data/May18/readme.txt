PEW RESEARCH CENTER FOR THE PEOPLE & THE PRESS
MAY RDD/RBS STUDY
April 25-May 1, 2018
N=1,503

***************************************************************************************************************************

NOTE: This project included two studies: 1) a random-digit-dial study (RDD) and 2) a registration-based study (RBS). This 
dataset includes only the interviews conducted for the RDD study.

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

May 3, 2018. "Trump Viewed Less Negatively on Issues, but Most Americans Are Critical of His Conduct" 
http://www.people-press.org/2018/05/03/trump-viewed-less-negatively-on-issues-but-most-americans-are-critical-of-his-conduct/

May 8, 2018. "Public Is Skeptical of the Iran Agreement – and Trump’s Handling of the Issue" 
http://www.people-press.org/2018/05/08/public-is-skeptical-of-the-iran-agreement-and-trumps-handling-of-the-issue/

May 10, 2018. "Public Supports U.S. Talks With North Korea; Many Doubt Whether Its Leaders Are ‘Serious’" 
http://www.people-press.org/2018/05/10/public-supports-u-s-talks-with-north-korea-many-doubt-whether-its-leaders-are-serious/

October 9, 2018. "Comparing Survey Sampling Strategies: Random-Digit Dial vs. Voter Files"
http://www.pewresearch.org/methods/2018/10/09/comparing-survey-sampling-strategies-random-digit-dial-vs-voter-files/



Blog posts from this survey:

May 10, 2018. "Americans are generally positive about free trade agreements, more critical of tariff increases" 
http://www.pewresearch.org/fact-tank/2018/05/10/americans-are-generally-positive-about-free-trade-agreements-more-critical-of-tariff-increases/

May 16, 2018. "Democrats, Republicans give their parties so-so ratings for standing up for ‘traditional’ positions" 
http://www.pewresearch.org/fact-tank/2018/05/16/democrats-republicans-give-their-parties-so-so-ratings-for-standing-up-for-traditional-positions/

May 24, 2018. "Republicans turn more negative toward refugees as number admitted to U.S. plummets" 
http://www.pewresearch.org/fact-tank/2018/05/24/republicans-turn-more-negative-toward-refugees-as-number-admitted-to-u-s-plummets/

June 5, 2018. "More Americans view long-term decline in union membership negatively than positively" 
http://www.pewresearch.org/fact-tank/2018/06/05/more-americans-view-long-term-decline-in-union-membership-negatively-than-positively/

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