// Run with: mongosh --quiet mongodb://localhost:27017/bda_practical5 mongodb_crud.js

const students = db.getSiblingDB("bda_practical5").students;

const sampleDocuments = [
  {
    student_id: "24DIT001",
    name: "Aarav Shah",
    department: "Data Engineering",
    year: 2,
    semester: 4,
    cgpa: 8.7,
    skills: ["Python", "MongoDB", "Hadoop"],
    city: "Ahmedabad"
  },
  {
    student_id: "24DIT002",
    name: "Diya Patel",
    department: "Data Engineering",
    year: 2,
    semester: 4,
    cgpa: 9.1,
    skills: ["Python", "Spark", "SQL"],
    city: "Vadodara"
  },
  {
    student_id: "24DIT003",
    name: "Kabir Mehta",
    department: "Computer Engineering",
    year: 2,
    semester: 4,
    cgpa: 7.9,
    skills: ["Java", "MongoDB", "Docker"],
    city: "Surat"
  },
  {
    student_id: "24DIT004",
    name: "Myra Desai",
    department: "Computer Engineering",
    year: 3,
    semester: 6,
    cgpa: 8.4,
    skills: ["Python", "Spark", "AWS"],
    city: "Ahmedabad"
  }
];

print("Database: bda_practical5 | Collection: students");
print("\nCREATE: insertOne");
print("Command: db.students.deleteMany({})");
students.deleteMany({});
print("Command: db.students.insertOne(" + JSON.stringify(sampleDocuments[0]) + ")");
printjson(students.insertOne(sampleDocuments[0]));
print("Output:");
students.find({ student_id: "24DIT001" }, { _id: 0 }).forEach(printjson);

print("\nCREATE: insertMany");
print("Command: db.students.insertMany(" + JSON.stringify(sampleDocuments.slice(1)) + ")");
printjson(students.insertMany(sampleDocuments.slice(1)));
print("Output:");
students.find({}, { _id: 0 }).sort({ student_id: 1 }).forEach(printjson);

print("\nREAD: all students");
print('Command: db.students.find({}, { _id: 0 }).sort({ cgpa: -1 })');
print("Output:");
students.find({}, { _id: 0 }).sort({ cgpa: -1 }).forEach(printjson);

print("\nREAD: CGPA >= 8.5");
print('Command: db.students.find({ cgpa: { $gte: 8.5 } }, { _id: 0, student_id: 1, name: 1, cgpa: 1 })');
print("Output:");
students.find(
  { cgpa: { $gte: 8.5 } },
  { _id: 0, student_id: 1, name: 1, cgpa: 1 }
).forEach(printjson);

print("\nREAD: students with MongoDB skill");
print('Command: db.students.find({ skills: "MongoDB" }, { _id: 0, student_id: 1, name: 1, skills: 1 })');
print("Output:");
students.find(
  { skills: "MongoDB" },
  { _id: 0, student_id: 1, name: 1, skills: 1 }
).forEach(printjson);

print("\nREAD: year 2 and CGPA > 8");
print('Command: db.students.find({ year: 2, cgpa: { $gt: 8 } }, { _id: 0, student_id: 1, name: 1, year: 1, cgpa: 1 })');
print("Output:");
students.find(
  { year: 2, cgpa: { $gt: 8 } },
  { _id: 0, student_id: 1, name: 1, year: 1, cgpa: 1 }
).forEach(printjson);

print("\nUPDATE: one document");
print('Command: db.students.updateOne({ student_id: "24DIT001" }, { $set: { cgpa: 8.9 }, $addToSet: { skills: "PyMongo" } })');
printjson(students.updateOne(
  { student_id: "24DIT001" },
  { $set: { cgpa: 8.9 }, $addToSet: { skills: "PyMongo" } }
));
print("Output:");
students.find({ student_id: "24DIT001" }, { _id: 0 }).forEach(printjson);

print("\nUPDATE: many documents");
print('Command: db.students.updateMany({ department: "Computer Engineering" }, { $set: { active: true } })');
printjson(students.updateMany(
  { department: "Computer Engineering" },
  { $set: { active: true } }
));
print("Output:");
students.find({ department: "Computer Engineering" }, { _id: 0 }).forEach(printjson);

print("\nINDEX: query statistics before and after city index");
if (students.getIndexes().some((index) => index.name === "city_1")) {
  print('Command: db.students.dropIndex("city_1")');
  students.dropIndex("city_1");
}
print('Command: db.students.find({ city: "Ahmedabad" }).explain("executionStats")');
const beforeIndex = students.find({ city: "Ahmedabad" }).explain("executionStats");
print("Command: db.students.createIndex({ city: 1 })");
students.createIndex({ city: 1 });
print('Command: db.students.find({ city: "Ahmedabad" }).explain("executionStats")');
const afterIndex = students.find({ city: "Ahmedabad" }).explain("executionStats");
print("Output:");
printjson({
  index: "city_1",
  documentsExaminedBefore: beforeIndex.executionStats.totalDocsExamined,
  documentsExaminedAfter: afterIndex.executionStats.totalDocsExamined,
  executionTimeMillisBefore: beforeIndex.executionStats.executionTimeMillis,
  executionTimeMillisAfter: afterIndex.executionStats.executionTimeMillis
});

print("\nAGGREGATION: department summary");
print('Command: db.students.aggregate([{ $group: { _id: "$department", student_count: { $sum: 1 }, average_cgpa: { $avg: "$cgpa" }, highest_cgpa: { $max: "$cgpa" } } }, { $sort: { average_cgpa: -1 } }, { $project: { _id: 0, department: "$_id", student_count: 1, average_cgpa: { $round: ["$average_cgpa", 2] }, highest_cgpa: 1 } }])');
print("Output:");
students.aggregate([
  {
    $group: {
      _id: "$department",
      student_count: { $sum: 1 },
      average_cgpa: { $avg: "$cgpa" },
      highest_cgpa: { $max: "$cgpa" }
    }
  },
  { $sort: { average_cgpa: -1 } },
  {
    $project: {
      _id: 0,
      department: "$_id",
      student_count: 1,
      average_cgpa: { $round: ["$average_cgpa", 2] },
      highest_cgpa: 1
    }
  }
]).forEach(printjson);

print("\nDELETE: one document");
print('Command: db.students.deleteOne({ student_id: "24DIT004" })');
printjson(students.deleteOne({ student_id: "24DIT004" }));
print("Output:");
students.find({}, { _id: 0 }).sort({ student_id: 1 }).forEach(printjson);

print("\nDELETE: documents with CGPA < 8");
print('Command: db.students.deleteMany({ cgpa: { $lt: 8 } })');
printjson(students.deleteMany({ cgpa: { $lt: 8 } }));
print("Output:");
students.find({}, { _id: 0 }).sort({ student_id: 1 }).forEach(printjson);

print("\nRemaining documents: " + students.countDocuments({}));
print("Command to delete the collection: db.students.drop()");