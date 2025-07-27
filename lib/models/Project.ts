import mongoose, { Document, Schema } from 'mongoose';

export interface IProject extends Document {
  name: string;
  description?: string;
  // Add other project properties here
}

const ProjectSchema: Schema = new Schema({
  name: { type: String, required: true },
  description: { type: String },
  // Define other project properties here
});

const Project = mongoose.models.Project || mongoose.model<IProject>('Project', ProjectSchema);

export default Project;
