# Homework 1: Designing a Metadata Schema
## Due: 9/27/2022 @ 5 pm

### Background

The value of data is determined by its Findability, Accessibility, Interoperability, and Reusability (FAIR) aspects. The FAIRness of data is largely determined by the (meta)data and schema. In your research, labs, or company you have all collected data. For this assignment select a data source from your work and construct a (meta)data schema. If you do not have any sufficient data from your work you can find data from an online repository (e.g. Zenodo, OSF, the materials project, etc.) to use for this assignment.

### Assignment

For this assignment design a schema that contains the Dublin Core Elements (see below) and at least 10 additional (meta)data catagories. For each of these catagories give a name and explain the use of the metadata category. Using two data records fill out the (meta)data for these records.  

#### Dublin Core Metadata Standard

<p>Built into the Dublin Core standard are definitions of each metadata element &ndash; like native content standard &ndash; that state what kinds of information should be recorded where and how.&nbsp; Associated with many of the data elements are data value standards such as the DCMI Type Vocabulary and ISO 639 language codes, etc. More information can be found on the <a href="https://www.dublincore.org/specifications/dublin-core/dcmi-terms/">Dublin Core Metadata Initiative website</a>.</p>

<table border="1" bordercolor="#ccc" cellpadding="5" cellspacing="0" class="table table-condensed" height="505" style="border-collapse:collapse;" width="766">
	<thead>
		<tr>
			<th class="ck_border" scope="col"><strong>Dublin Core Element</strong></th>
			<th class="ck_border" scope="col"><strong>Use</strong></th>
			<th class="ck_border" scope="col"><strong>Possible Data Value Standards</strong></th>
		</tr>
	</thead>
	<tbody>
		<tr>
			<td class="ck_border"><strong>Title</strong></td>
			<td class="ck_border">A name given to the resource.</td>
			<td class="ck_border">&nbsp;</td>
		</tr>
		<tr>
			<td class="ck_border"><strong>Subject</strong></td>
			<td class="ck_border">The topic of the resource.</td>
			<td class="ck_border"><a href="http://authorities.loc.gov/">Library of Congress Subject Headings (LCSH)</a></td>
		</tr>
		<tr>
			<td class="ck_border"><strong>Description</strong></td>
			<td class="ck_border">An account of the resource.</td>
			<td class="ck_border">&nbsp;</td>
		</tr>
		<tr>
			<td class="ck_border"><strong>Creator</strong></td>
			<td class="ck_border">An entity primarily responsible for making the resource.</td>
			<td class="ck_border"><a href="http://authorities.loc.gov/">Library of Congress Name Authority File (LCNAF)</a></td>
		</tr>
		<tr>
			<td class="ck_border"><strong>Publisher</strong></td>
			<td class="ck_border">An entity responsible for making the resource available.</td>
			<td class="ck_border">&nbsp;</td>
		</tr>
		<tr>
			<td class="ck_border"><strong>Contributor</strong></td>
			<td class="ck_border">An entity responsible for making contributions to the resource.</td>
			<td class="ck_border"><a href="http://authorities.loc.gov/">Library of Congress Name Authority File (LCNAF)</a></td>
		</tr>
		<tr>
			<td class="ck_border"><strong>Date</strong></td>
			<td class="ck_border">A point or period of time associated with an event in the lifecycle of the resource.</td>
			<td class="ck_border"><a href="http://www.w3.org/TR/NOTE-datetime">W3CDTF</a></td>
		</tr>
		<tr>
			<td class="ck_border"><strong>Type</strong></td>
			<td class="ck_border">The nature or genre of the resource.</td>
			<td class="ck_border"><a href="https://www.dublincore.org/specifications/dublin-core/dcmi-terms/#section-7">DCMI Type Vocabulary</a></td>
		</tr>
		<tr>
			<td class="ck_border"><strong>Format</strong></td>
			<td class="ck_border">The file format, physical medium, or dimensions of the resource.</td>
			<td class="ck_border"><a href="http://www.iana.org/assignments/media-types/">Internet Media Types (MIME)</a></td>
		</tr>
		<tr>
			<td class="ck_border"><strong>Identifier</strong></td>
			<td class="ck_border">An unambiguous reference to the resource within a given context.</td>
			<td class="ck_border">&nbsp;</td>
		</tr>
		<tr>
			<td class="ck_border"><strong>Source</strong></td>
			<td class="ck_border">A related resource from which the described resource is derived.</td>
			<td class="ck_border">&nbsp;</td>
		</tr>
		<tr>
			<td class="ck_border"><strong>Language</strong></td>
			<td class="ck_border">A language of the resource.</td>
			<td class="ck_border"><a href="https://www.loc.gov/standards/iso639-2/php/code_list.php">ISO 639</a></td>
		</tr>
		<tr>
			<td class="ck_border"><strong>Relation</strong></td>
			<td class="ck_border">A related resource.</td>
			<td class="ck_border">&nbsp;</td>
		</tr>
		<tr>
			<td class="ck_border"><strong>Coverage</strong></td>
			<td class="ck_border">The spatial or temporal topic of the resource, the spatial applicability of the resource, or the jurisdiction under which the resource is relevant.</td>
			<td class="ck_border"><a href="http://www.getty.edu/research/tools/vocabulary/tgn/index.html">Thesaurus of Geographic Names (TGN)</a></td>
		</tr>
		<tr>
			<td class="ck_border"><strong>Rights</strong></td>
			<td class="ck_border">Information about rights held in and over the resource.</td>
			<td class="ck_border">&nbsp;</td>
		</tr>
	</tbody>
</table>

### Submission Instructions

The data records and schema should be submitted as an Excel Spreadsheet. If you feel comfortable using python, you could instead use a dictionary or JSON format for the submission. 