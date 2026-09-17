# Import the JSON array into MongoDB, then run analytical queries.
# Run from the repository root:
# powershell -ExecutionPolicy Bypass -File Practical5\import_and_analyze.ps1

$database = "bda_practical5"
$collection = "students_imported"
$jsonFile = "Practical5\students.json"

Write-Host "IMPORT: students.json"
Write-Host "Command: mongoimport --db $database --collection $collection --file $jsonFile --jsonArray --drop"
$mongoimport = Get-Command mongoimport -ErrorAction SilentlyContinue
if ($null -ne $mongoimport) {
  & mongoimport --db $database --collection $collection --file $jsonFile --jsonArray --drop
  if ($LASTEXITCODE -ne 0) {
    throw "mongoimport failed."
  }
} else {
  Write-Host "mongoimport was not found; using mongosh CLI JSON fallback."
  $rawJson = Get-Content -Raw $jsonFile
  $jsonBase64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($rawJson))
  $importCommand = "const documents = JSON.parse(Buffer.from('$jsonBase64', 'base64').toString()); db.getSiblingDB('$database').$collection.drop(); db.getSiblingDB('$database').$collection.insertMany(documents);"
  Write-Host "Command: mongosh --quiet --eval `"$importCommand`""
  & mongosh --quiet --eval $importCommand
  if ($LASTEXITCODE -ne 0) {
    throw "mongosh JSON import failed."
  }
}

Write-Host "ANALYSIS: imported dataset"
Write-Host "Command: mongosh $database --eval ..."
& mongosh --quiet $database --eval @'
const students = db.students_imported;

print("\nQuery 1: total number of imported students");
print("Command: db.students_imported.countDocuments({})");
print("Output: " + students.countDocuments({}));

print("\nQuery 2: average, minimum, and maximum CGPA");
print('Command: db.students_imported.aggregate([{ $group: { _id: null, average_cgpa: { $avg: "$cgpa" }, minimum_cgpa: { $min: "$cgpa" }, maximum_cgpa: { $max: "$cgpa" } } }])');
students.aggregate([
  {
    $group: {
      _id: null,
      average_cgpa: { $avg: "$cgpa" },
      minimum_cgpa: { $min: "$cgpa" },
      maximum_cgpa: { $max: "$cgpa" }
    }
  },
  {
    $project: {
      _id: 0,
      average_cgpa: { $round: ["$average_cgpa", 2] },
      minimum_cgpa: 1,
      maximum_cgpa: 1
    }
  }
]).forEach(printjson);

print("\nQuery 3: department-wise student count and average CGPA");
print('Command: db.students_imported.aggregate([{ $group: { _id: "$department", student_count: { $sum: 1 }, average_cgpa: { $avg: "$cgpa" } } }, { $sort: { average_cgpa: -1 } }])');
students.aggregate([
  {
    $group: {
      _id: "$department",
      student_count: { $sum: 1 },
      average_cgpa: { $avg: "$cgpa" }
    }
  },
  { $sort: { average_cgpa: -1 } },
  {
    $project: {
      _id: 0,
      department: "$_id",
      student_count: 1,
      average_cgpa: { $round: ["$average_cgpa", 2] }
    }
  }
]).forEach(printjson);

print("\nQuery 4: students with CGPA >= 8.5");
print('Command: db.students_imported.find({ cgpa: { $gte: 8.5 } }, { _id: 0, name: 1, department: 1, cgpa: 1 }).sort({ cgpa: -1 })');
students.find(
  { cgpa: { $gte: 8.5 } },
  { _id: 0, name: 1, department: 1, cgpa: 1 }
).sort({ cgpa: -1 }).forEach(printjson);

print("\nQuery 5: count students by city");
print('Command: db.students_imported.aggregate([{ $group: { _id: "$city", student_count: { $sum: 1 } } }, { $sort: { student_count: -1 } }])');
students.aggregate([
  { $group: { _id: "$city", student_count: { $sum: 1 } } },
  { $sort: { student_count: -1 } },
  { $project: { _id: 0, city: "$_id", student_count: 1 } }
]).forEach(printjson);
'@
if ($LASTEXITCODE -ne 0) {
    throw "mongosh analytical queries failed."
}
