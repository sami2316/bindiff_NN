"""
BinDiff_NN: Learning Distributed Representation of Assembly for Robust
            Binary Diffing against Semantic Differences
Copyright (c) 2020-2021, Sami Ullah

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import sqlite3
import os, sys
import csv
import re
import random
import hashlib
from os import walk
from io import StringIO
from difflib import SequenceMatcher

#-----------------------------------------------------------------------
def result_iter(cursor, arraysize=1000):
  'An iterator that uses fetchmany to keep memory usage down'
  while True:
    results = cursor.fetchmany(arraysize)
    if not results:
      break
    for result in results:
      yield result

"""
    Get similarity ratio between two text codes
"""
def real_quick_ratio(buf1, buf2):
  try:
    if buf1 is None or buf2 is None or buf1 == "" or buf1 == "":
      return 0
    s = SequenceMatcher(None, buf1, buf2)
    return s.real_quick_ratio()
  except:
    print("real_quick_ratio:", str(sys.exc_info()[1]))
    return 0
#-----------------------------------------------------------------------
CMP_REPS = ['loc_FFFF', 'locret_FFFF', 'loc_ffff', 'locret_ffff','j_nulsub',
            'j_sub_', 'sub_', 'qword_', 'dword_', 'byte_', 'word_', 'off_',
            'def_', 'unk_', 'asc_', 'stru_', 'dbl_']
CMP_REMS = ["dword ptr ", "byte ptr ", "qword ptr ", "word ptr ", "short ptr "]
#-----------------------------------------------------------------
'''
  Manage Database Connections
'''
class Database:
  def __init__(self, db_name):
    self.db_name = db_name
    self.db = None

  '''
    Open Connection to the Database
  '''
  def open_db(self):
    try:
      db = sqlite3.connect(self.db_name)
      db.text_factory = str
      db.row_factory = sqlite3.Row
      self.db = db
    except:
      print("The Database doesn't exist...")

  '''
    Get the DB cursor
  '''
  def db_cursor(self):
    if self.db is None:
      self.open_db()

    return self.db.cursor()

  '''
  Attach second database
  '''
  def attach(self, db2):
    cur = self.db_cursor()
    cur.execute('attach "%s" as diff' % db2)
    cur.close()

  '''
    Close the DB connection
  '''
  def db_close(self):
    if self.db is not None:
      self.db.close()

#-----------------------------------------------------------------
# Data corresponding to a single function
class AsmData(object):
  def __init__(self):
    self.label = None
    self.instructions_a = []
    self.instructions_b = []
#-----------------------------------------------------------------
class ASMParser(Database):
  def __init__(self, dirpath):
    self.dir_path = os.path.join(os.getcwd(), dirpath)
    self.dbfile_list = []
    self.csvfile_list = []
    self.functions = []
    self.re_cache = {}
    self.hashes = []

  def list_files(self, directory):
    files = []
    for (dirpath, dirnames,filenames) in walk(directory):
      files.extend(filenames)
      break
    for l in files:
      tmp = l.split(".")
      if tmp[-1] == "sqlite":
        self.dbfile_list.append(l)
      elif tmp[-1] == 'csv':
        self.csvfile_list.append(l)

  # Get clean Assembly
  def get_cmp_asm_lines(self, asm):
    sio = StringIO(asm)
    lines = []
    get_cmp_asm = self.get_cmp_asm
    for line in sio.readlines():
      line = line.strip("\n")
      lines.append(get_cmp_asm(line))
    return "\n".join(lines) + "\n"

  def get_cmp_asm(self, asm):
    if asm is None:
      return asm

    #ignore the comments in the assembly dump
    tmp = asm.split(";")[0]
    tmp = tmp.split("#")[0]

    # delete any sub_, loc_XXXXXXXX etc
    for rep in CMP_REPS:
      tmp = self.re_sub(rep + '[a-f0-9A-F]+', 'XXXX', tmp)

    for rep in CMP_REMS:
      tmp = self.re_sub(rep ,'', tmp)

    reps = ['\+[a-f0-9A-F]+h+', '\-[a-f0-9A-F]+h+', 'cs:[A-F0-9]+', ' [a-f0-9A-F]+h+']
    tmp = self.re_sub(reps[0], '+XXXXh', tmp)
    tmp = self.re_sub(reps[1], '-XXXXh', tmp)
    tmp = self.re_sub(reps[2], 'cs:', tmp)
    tmp = self.re_sub(reps[3], ' XXXXh', tmp)
    tmp = self.re_sub('\.\.[a-f0-9A-F]{8}', 'XXX', tmp)

    # strip any possible remaining white-space character
    tmp = self.re_sub('[[ \t\n]+$', '', tmp)

    return tmp

  def re_sub(self, text, repl, string):
    if text not in self.re_cache:
      self.re_cache[text] = re.compile(text, flags=re.IGNORECASE)

    re_obj = self.re_cache[text]
    return re_obj.sub(repl, string)

  def parse_one(self, file_name):
    # read the csv file
    partial = []
    count = 0
    csv_path = os.path.join(self.dir_path, file_name)
    with open(csv_path) as fp:
      csv_reader = csv.reader(fp, delimiter=',')
      if 'xnu-4570_41_51' in file_name:
        partial = [line[0] for line in csv_reader][1:]
      else:
        partial = ['_'+line[0] if '::' not in line[0] else line[0] for line in csv_reader][1:]

    tmp = file_name.split('_')
    db1 = tmp[0] + '_' + tmp[1][:2] + '.sqlite'
    db2 = tmp[0] + '_' + tmp[2][:2] + '.sqlite'
    if db1 not in self.dbfile_list and db2 not in self.dbfile_list:
      return

    db1_path = os.path.join(self.dir_path, db1)
    db2_path = os.path.join(self.dir_path, db2)
    db_handle = Database(db1_path)
    db_handle.attach(db2_path)
    print("Extracting the assembly from {0} and {1}".format(db1, db2))
    query_assembly = """select f.name name,f.assembly asm1, df.assembly asm2 from
                        functions f, diff.functions df where f.name == df.name"""

    cur = db_handle.db_cursor()
    if cur is None:
      return

    cur.execute(query_assembly)
    # iterate over the assembly functions and collect unique functions
    lfunctions = {}
    for row in result_iter(cur):
      func = AsmData()
      if any(s == row['name'] for s in partial):
        func.label = 'partial'
      else:
        func.label = 'match'
      func.instructions_a = self.get_cmp_asm_lines(row['asm1'])
      func.instructions_b = self.get_cmp_asm_lines(row['asm2'])
      hash = hashlib.md5(str(func.instructions_a).encode() + str(func.instructions_b).encode()).hexdigest()

      if hash in self.hashes:
        continue
      lfunctions[count] = func
      self.hashes.append(hash)
      count += 1


    # prepare the partial samples
    par_functions = {}
    for idx, item in lfunctions.items():

      func = AsmData()
      func.label = 'partial'
      func.instructions_a = item.instructions_a
      tmp = item.instructions_b.split('XXXX:\n')
      if len(tmp) <= 2:
        continue

      
      # 0. randomnly split and select the b_index basic block
      b_index = random.randrange(1, len(tmp)-1,1)
      shuffled = tmp[b_index].split('\n')
      shuffled.remove('')
      if len(shuffled) < 4:
        continue

      # 1. keep 70% instructions
      #split_ratio = int(round(0.80*len(shuffled))) 
      #i_blk = '\n'.join(shuffled[:split_ratio])+'\n'
      #func.instructions_b = "XXXX:\n".join(tmp[0:b_index] + [i_blk] + tmp[b_index+1:])

      # 2. Randomly skip 2 instructions from the b_index basic block
      #i_index = random.randrange(1,len(shuffled)-2,1)
      #i_blk = '\n'.join(shuffled[0:i_index] + shuffled[i_index+2:]) + '\n'
      #func.instructions_b = "XXXX:\n".join(tmp[0:b_index] + [i_blk] + tmp[b_index+1:])

      # 3. randomly skip 1 basic block
      func.instructions_b = "XXXX:\n".join(tmp[0:b_index] + tmp[b_index+1:])
      hash = hashlib.md5(str(func.instructions_a).encode() + str(func.instructions_b).encode()).hexdigest()

      if hash in self.hashes:
        continue
      par_functions[count] = func
      count += 1
      

    cur.close()
    db_handle.db_close()
    self.functions.extend(list(lfunctions.values()))
    self.functions.extend(list(par_functions.values()))
    #self.functions.extend(list(diff_functions.values()))

  def parse_all(self):
    self.list_files(self.dir_path)

    for filename in self.csvfile_list:
      self.parse_one(filename)

    with open('asmdata/assembly_data.txt', 'w') as fp:
      for item in self.functions:
        fp.write('label:' + str(item.label) + '\n')
        fp.write('assembly_a:\n' + item.instructions_a)
        fp.write('assembly_b:\n' + item.instructions_b)
        fp.write('\n')

if __name__ == '__main__':
  handle = ASMParser('databases')
  handle.parse_all()
