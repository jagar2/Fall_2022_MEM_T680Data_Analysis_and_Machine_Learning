# Final Project

## Goals

This project aims to apply the tools and concepts taught in this course to a problem that interests you. This project is intentionally open-ended. Example projects might include: building a data management workflow for a process in your company or research, developing an interactive dashboard and data visualization tool, or developing an automated analysis pipeline for a task of interest. The goal is for you to get something useful out of this project while demonstrating the ability to apply computational tools. It can be an approved topic if you justify how your project fits into the course. Note that it is not a requirement that you complete the project within the timeline of the course. You will be evaluated on the progress that you have made, the use of computational tools, and the communication of impact and tasks. 

Most importantly, I cannot teach you in a single course how to be proficient in applying modern computational methods. The most important skill you can get from this course is the ability and confidence to approach problems with computational methods. It is expected for this project that you will extend beyond the topics covered in the course.

## White Paper
**Due 10/12/2022**

To determine the topic for your project, you will submit a one-page white paper and a logical diagram of your proposed project. This white paper will not be graded. Instead, it will be used to make sure you are on the right track for your project. This white paper should describe:
1. The current problem you are trying to address
1. What is the current process that is used to solve the problem
1. Your approach to solving the problem. *Note it is okay if you do not know the exact methodology that you will use. This will be the primary area of our discussion*
1. Timeline for project completion

## Assignment Components

### GitHub Repository (50%)
For your assignment, you are required to create and manage your software development using version control on GitHub. Your repository does not have to be public but will need to be shared for grading. The GitHub Repository will be evaluated using the following rubric.

#### GitHub Repository Rubric

Grading will be completed through peer evaluations and evaluations by Prof. Agar. Each student will be responsible for evaluating 3-4 classmates' projects. At Prof. Agar's discretion, he can exclude any peer evaluation, which he feels is unfair.

<html>
<style>
table, th, td, tr {
  border: 1px solid black;
}
</style>
<table class="center" style=100 width="100%">
    <style>
    tr:nth-child(even) {
    background-color: #FFFFFF;}
    border: 1px solid black;
    border-collapse: collapse;
    </style>
  <tr>
    <th style = "text-align:center">Topic</th>
    <th style = "text-align:center">Percentage</th>
  </tr>
  <tr>
    <td style = "text-align:center">Code Functionality - <i>How much functionality is included in the code?</i></td>
    <td style = "text-align:center">40</td>
  </tr>
  <tr>
    <td style = "text-align:center">In code documentation - <i>How readable is your code?</i></td>
    <td style = "text-align:center">15</td>
  </tr>
  <tr>
    <td style = "text-align:center">Code Reusability - <i>How easy is it for someone to reuse and extend your code?</i> </td>
    <td style = "text-align:center">25</td>
  </tr>
  <tr>
    <td style = "text-align:center">Usage Documentation - <i> How easy is it for someone new to your code to use it?</i> </td>
    <td style = "text-align:center">20</td>
  </tr>
</table>
</html>

#### Hints

* Make sure to comment on most lines of code
* Functions can be documented using a standard format for autodoc string
* Try to follow python conventions PEP8 or Black
* Make tutorial notebooks that users can explore
* Create a Jupyterbook or readthedocs
* Add environment configuration file
* Serve your package in a Docker container
* Add a license file for reuse
* Add continuous integration

Note that you do not need to fulfill all of these hints. I would be shocked if you did more than half of them. Please make sure you do what is meaningful for your project.  

### Presentation (50%)

You will be required to prepare a 12-minute presentation describing your final project. The project will be graded using a combination of peer reviews and Prof. Agar's evaluation. At Prof. Agar's discretion, he can exclude any peer evaluation, which he feels is unfair.

#### Topics to Cover:
1. The problem that your project addresses
1. Computational tools which are developed
1. Demonstration of how your tool works
1. Future development roadmap and impact

#### Presentation Rubric

<table class="center" style=100 width="100%">
    <style>
    tr:nth-child(even) {
    background-color: #FFFFFF;}
    border: 1px solid black;
    border-collapse: collapse;
    </style>
    <colgroup>
       <col span="1" style="width: 15%;">
       <col span="1" style="width: 75%;">
       <col span="1" style="width: 5%;">
       <col span="1" style="width: 5%;">
    </colgroup>
  <tr>
    <th style = "text-align:center">Category</th>
    <th style = "text-align:center">Scoring Criteria</th>
    <th style = "text-align:center">Total Points</th>
    <th style = "text-align:center">Points Scored</th>
  </tr>
  <tr>
    <td rowspan="3" style = "text-align:center"> <b>Organization (15 points)</b></td>
    <td style = "text-align:center">The type of presentation is appropriate for the topic and audience. </td>
    <td style = "text-align:center">5</td>
    <td>&nbsp;</td>
  </tr>
  <tr>
    <td style = "text-align:center">Information is presented in a logical sequence.</td>
    <td style = "text-align:center">5</td>
    <td>&nbsp;</td>
  </tr>
  <tr>
    <td style = "text-align:center">Presentation includes appropriate citations </td>
    <td style = "text-align:center">5</td>
    <td>&nbsp;</td>
  </tr>
  <tr>
    <td rowspan="6" style = "text-align:center"><b>Content (45 points) </b></td>
    <td style = "text-align:center">Introduction is attention-getting, establishes the problem well, and establishes a framework for the rest of the presentation.</td>
    <td style = "text-align:center">5</td>
    <td>&nbsp;</td>
  </tr>
    <tr>
    <td style = "text-align:center"> Technical terms are well-defined in language appropriate for the target audience. </td>
    <td style = "text-align:center">5</td>
    <td>&nbsp;</td>
  </tr>
    <tr>
    <td style = "text-align:center"> Presentation contains accurate information.  </td>
    <td style = "text-align:center">10</td>
    <td>&nbsp;</td>
  </tr>
    <tr>
    <td style = "text-align:center"> Material included is relevant to the overall Message/purpose.
    </td>
    <td style = "text-align:center">10</td>
    <td>&nbsp;</td>
  </tr>
      <tr>
    <td style = "text-align:center"> Appropriate amount of material is prepared, and points made reflect well their relative importance.
    </td>
    <td style = "text-align:center">10</td>
    <td>&nbsp;</td>
  </tr>
    <tr>
    <td style = "text-align:center"> There is an obvious conclusion summarizing the presentation.
    </td>
    <td style = "text-align:center">5</td>
    <td>&nbsp;</td>
  </tr>
    <tr>
    <td rowspan="6" style = "text-align:center"> <b>Presentation (40 points)</b>
    </td>
    <td style = "text-align:center"> Speakers maintains good eye contact with the
audience and is appropriately animated (e.g.,
gestures, moving around, etc.)</td>
    <td style = "text-align:center">5</td>
    <td>&nbsp;</td>
  </tr>
      <tr>
    <td style = "text-align:center"> Speaker uses a clear, audible voice.
    </td>
    <td style = "text-align:center">5</td>
    <td>&nbsp;</td>
  </tr>
    </tr>
      <tr>
    <td style = "text-align:center"> Delivery is poised, controlled, and smooth.
    </td>
    <td style = "text-align:center">5</td>
    <td>&nbsp;</td>
  </tr>
    <tr>
    <td style = "text-align:center"> Visual aids are well prepared, informative,
effective, and not distracting
    </td>
    <td style = "text-align:center">10</td>
    <td>&nbsp;</td>
  </tr>
      <tr>
    <td style = "text-align:center"> Length of presentation is within the assigned
time limits.
    </td>
    <td style = "text-align:center">5</td>
    <td>&nbsp;</td>
  </tr>
        <tr>
    <td style = "text-align:center"> Information was well communicated.
    </td>
    <td style = "text-align:center">10</td>
    <td>&nbsp;</td>
  </tr>
    <tr>
    <td style = "text-align:center"> <b> Score </b>
    </td>
    <td style = "text-align:center"> <b> Total </b>
    </td>
    <td style = "text-align:center"> <b> 100 </b>
    </td>
    <td>&nbsp;</td>
  </tr>
</table>
